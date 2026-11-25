"""Vision-language models through transformers: batched, greedy, zero-shot."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

from ..models import Prediction
from ..preprocess import upscale_longest_edge

LOOP_WINDOW = 256  # generated tokens that must be periodic before a page is stopped
LOOP_PERIOD = 64  # longest period that counts as a loop
LOOP_EVERY = 32  # steps between checks


def is_looping(tokens: Sequence[int]) -> bool:
    """True if the last LOOP_WINDOW tokens repeat a pattern of at most LOOP_PERIOD tokens."""
    if len(tokens) < LOOP_WINDOW:
        return False
    tail = tokens[-LOOP_WINDOW:]
    return any(
        all(tail[i] == tail[i - period] for i in range(period, LOOP_WINDOW))
        for period in range(1, LOOP_PERIOD + 1)
    )


def cut_at_end(tokens: Sequence[int], end_tokens: set[int]) -> tuple[list[int], bool]:
    """(tokens before the first end token, whether an end token was present)."""
    for i, token in enumerate(tokens):
        if token in end_tokens:
            return list(tokens[:i]), True
    return list(tokens), False


def encode_images_separately(model) -> None:
    """Make `model.get_image_features` run the vision encoder on one image at a time.

    LightOnOCR's encoder joins all images of a batch into one sequence and masks attention
    between them, so memory grows with the square of the batch size. The result is the same.
    """
    encode = model.get_image_features

    def get_image_features(pixel_values, image_sizes, **kwargs):
        outputs = [
            encode(pixel_values=pixel_values[i : i + 1], image_sizes=image_sizes[i : i + 1], **kwargs)
            for i in range(len(image_sizes))
        ]
        first = outputs[0]
        first.pooler_output = tuple(part for output in outputs for part in output.pooler_output)
        return first

    model.get_image_features = get_image_features


class LoopGuard:
    """Stops a page that repeats itself, so it does not hold up its whole batch.

    Passed to `generate` as a stopping criterion; transformers only needs a callable.
    """

    def __init__(self, prompt_length: int, batch_size: int, end_tokens: set[int]):
        self.prompt_length = prompt_length
        self.end_tokens = end_tokens
        self.stopped = [False] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        generated = input_ids.shape[1] - self.prompt_length
        if generated >= LOOP_WINDOW and generated % LOOP_EVERY == 0:
            for i, tail in enumerate(input_ids[:, -LOOP_WINDOW:].tolist()):
                # A finished page is padded with end tokens, which is not a loop.
                if not self.stopped[i] and tail[-1] not in self.end_tokens:
                    self.stopped[i] = is_looping(tail)
        return torch.tensor(self.stopped, device=input_ids.device, dtype=torch.bool)


class _VLM:
    def __init__(self, config: dict):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("A CUDA GPU is required.")
        self.config = config
        self.dtype = getattr(torch, config["dtype"])
        self.vision_dtype = getattr(torch, config.get("vision_dtype", config["dtype"]))
        self.processor, model = self._load(config, self.dtype)
        self.model = model.to("cuda").eval()
        self.processor.tokenizer.padding_side = "left"
        generation = self.model.generation_config
        ends = generation.eos_token_id
        self.end_tokens = set(ends if isinstance(ends, list) else [ends])
        # Padding after the end must also read as "finished".
        self.end_tokens.add(generation.pad_token_id)

    def _load(self, config: dict, dtype):
        raise NotImplementedError

    def _content(self) -> list[dict]:
        return [{"type": "image"}]

    def transcribe(self, image_paths: Sequence[Path]) -> list[Prediction]:
        import torch

        try:
            with torch.inference_mode():
                return self._generate(image_paths)
        except torch.cuda.OutOfMemoryError:
            pass
        # Outside the handler, so that the failed batch's tensors are no longer referenced.
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(f"CUDA out of memory with {len(image_paths)} images")

    def _generate(self, image_paths: Sequence[Path]) -> list[Prediction]:
        from PIL import Image

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                page = image.convert("RGB")
            images.append(upscale_longest_edge(page, self.config.get("upscale_to", 0)))
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": self._content()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
            images_kwargs={"size": dict(self.config["image_size"])},
        ).to("cuda", dtype=self.vision_dtype)
        inputs.pop("token_type_ids", None)
        prompt_length = inputs["input_ids"].shape[1]
        guard = LoopGuard(prompt_length, len(images), self.end_tokens)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            stopping_criteria=_criteria(guard),
        )
        predictions = []
        for row, looped in zip(output[:, prompt_length:].tolist(), guard.stopped, strict=True):
            tokens, finished = cut_at_end(row, self.end_tokens)
            text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            predictions.append(Prediction(text, len(tokens), truncated=looped or not finished))
        return predictions

    def info(self) -> dict:
        tensors = list(self.model.parameters())
        return {
            "device": "cuda",
            "dtype": self.config["dtype"],
            "vision_dtype": self.config.get("vision_dtype", self.config["dtype"]),
            "parameters": sum(t.numel() for t in tensors),
            "weights_mb": round(sum(t.numel() * t.element_size() for t in tensors) / 1e6, 1),
        }


def _criteria(guard: LoopGuard):
    from transformers import StoppingCriteriaList

    return StoppingCriteriaList([guard])


class LightOnModel(_VLM):
    def _load(self, config: dict, dtype):
        # The checkpoint's config says "mistral3", so Auto classes would pick the wrong model.
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        name, revision = config["hf_id"], config["revision"]
        processor = LightOnOcrProcessor.from_pretrained(name, revision=revision)
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            name, revision=revision, dtype=dtype, attn_implementation="sdpa"
        )
        # In float16 the encoder gave NaN on every page.
        model.model.vision_encoder.to(self.vision_dtype)
        model.model.vision_projection.to(self.vision_dtype)
        encode_images_separately(model.model)
        return processor, model


class QwenVLModel(_VLM):
    def _load(self, config: dict, dtype):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        name, revision = config["hf_id"], config["revision"]
        processor = AutoProcessor.from_pretrained(name, revision=revision)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            name, revision=revision, dtype=dtype, attn_implementation="sdpa"
        )
        return processor, model

    def _content(self) -> list[dict]:
        return [{"type": "image"}, {"type": "text", "text": self.config["prompt"]}]
