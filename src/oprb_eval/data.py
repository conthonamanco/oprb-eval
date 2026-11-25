"""OPRB test set: download, pairing, ground-truth parsing, sampling."""

from __future__ import annotations

import random
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "kpurkayastha/OPRB"
EXPECTED = {"text": 6000, "nontext": 3000}
SUFFIX = {"img": ".jpg", "gt": ".txt"}


@dataclass(frozen=True)
class Page:
    category: str  # "text" | "nontext"
    sample_id: str
    image_path: Path
    gt_path: Path
    occlusion_type: str
    density: str

    @property
    def key(self) -> str:
        # The same stem may exist in both categories, so the stem alone is not unique.
        return f"{self.category}/{self.sample_id}"


def parse_stem(stem: str) -> tuple[str, str]:
    """`White_Whitener__2p0pct__105.tar_...` -> ("White_Whitener", "2p0pct")."""
    parts = stem.split("__", 2)
    if len(parts) < 3:
        return parts[0], "unknown"
    return parts[0], parts[1]


def read_gt_tokens(path: Path) -> list[str]:
    """Token column of a GT file, in file order (same rule as the sample notebook)."""
    tokens = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 11 and fields[0].strip():
                tokens.append(fields[0].strip())
    return tokens


def collect_pages(root: Path, categories=("text", "nontext")) -> list[Page]:
    pages: list[Page] = []
    for category in categories:
        images = {p.stem: p for p in (root / category / "img").glob("*.jpg")}
        gts = {p.stem: p for p in (root / category / "gt").glob("*.txt")}
        unpaired = images.keys() ^ gts.keys()
        if unpaired:
            raise ValueError(
                f"{category}: {len(unpaired)} files without a partner, e.g. {sorted(unpaired)[0]}"
            )
        if not images:
            raise FileNotFoundError(f"No test pages under {root / category}. Run `download` first.")
        for stem in sorted(images):
            kind, density = parse_stem(stem)
            pages.append(Page(category, stem, images[stem], gts[stem], kind, density))
    return pages


def require_full_test_set(pages: list[Page]) -> None:
    """Stop if a category does not have its exact number of pages (partial download)."""
    for category, count in Counter(p.category for p in pages).items():
        if count != EXPECTED[category]:
            raise SystemExit(f"{category}: found {count} pages, expected {EXPECTED[category]}.")


def stratified_sample(pages: list[Page], per_category: int, seed: int = 42) -> list[Page]:
    """Deterministic sample, spread evenly over (type, density) within each category.

    Strata are served round-robin, so rare strata are not lost in small samples, and a
    larger sample with the same seed contains the smaller one.
    """
    strata: dict[str, dict[tuple[str, str], list[Page]]] = defaultdict(lambda: defaultdict(list))
    for page in sorted(pages, key=lambda p: p.key):
        strata[page.category][(page.occlusion_type, page.density)].append(page)

    chosen: list[Page] = []
    for category, groups in strata.items():
        queues = []
        for name, members in sorted(groups.items()):
            random.Random(f"{seed}/{category}/{name}").shuffle(members)
            queues.append(members)
        picked: list[Page] = []
        while len(picked) < per_category and any(queues):
            for queue in queues:
                if queue and len(picked) < per_category:
                    picked.append(queue.pop())
        chosen.extend(picked)
    return sorted(chosen, key=lambda p: p.key)


def extract_flat(archive: Path, destination: Path, suffix: str) -> int:
    """Extract files by basename only, so archive-supplied paths are never trusted."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(suffix):
                continue
            target = destination / member.name.replace("\\", "/").rsplit("/", 1)[-1]
            # Compare sizes so that a file cut short by an interrupted run is redone.
            if not (target.exists() and target.stat().st_size == member.size):
                stream = tar.extractfile(member)
                if stream is None:
                    continue
                with stream, target.open("wb") as out:
                    shutil.copyfileobj(stream, out)
            count += 1
    return count


def train_images(root: Path, count: int) -> list[Path]:
    """The first `count` training images, streamed so the 3 GB archive is not downloaded."""
    folder = root / "train_images"
    folder.mkdir(parents=True, exist_ok=True)
    have = sorted(folder.glob("*.jpg"))
    if len(have) >= count:
        return have[:count]
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/train/text/img.tar.gz"
    found: list[Path] = []
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".jpg"):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue
            target = folder / member.name.replace("\\", "/").rsplit("/", 1)[-1]
            with stream, target.open("wb") as out:
                shutil.copyfileobj(stream, out)
            found.append(target)
            if len(found) == count:
                break
    return sorted(found)


def download(root: Path, categories=("text", "nontext")) -> None:
    """Fetch only test images and test GT. `img_gt/` and `annots/` are never downloaded."""
    from huggingface_hub import hf_hub_download

    for category in categories:
        for kind, suffix in SUFFIX.items():
            archive = hf_hub_download(
                REPO_ID, f"test/{category}/{kind}.tar.gz", repo_type="dataset"
            )
            n = extract_flat(Path(archive), root / category / kind, suffix)
            print(f"{category}/{kind}: {n} files (expected {EXPECTED[category]})")
