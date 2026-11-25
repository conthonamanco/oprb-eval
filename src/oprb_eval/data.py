"""OPRB test set: download, pairing, ground-truth parsing, sampling."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "kpurkayastha/OPRB"
EXPECTED = {"text": 6000, "nontext": 3000}
TRAIN_TEXT_PAGES = 17212
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


def _draw(*parts) -> float:
    """Pseudo-random number in [0, 1), the same on every Python version and machine."""
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:6], "big") / 2**48


def stratified_sample(pages: list[Page], total: int, seed: int = 42) -> list[Page]:
    """`total` random pages that keep the proportions of `pages`.

    Categories keep their ratio (2:1 in OPRB), and so do (type, density) groups inside each
    category, so plain means over the sample estimate means over all pages. Pages come in
    drawing order: a smaller sample with the same seed is a prefix of a larger one.
    """
    strata: dict[str, dict[tuple[str, str], list[Page]]] = defaultdict(lambda: defaultdict(list))
    for page in pages:
        strata[page.category][(page.occlusion_type, page.density)].append(page)

    ranked: list[tuple[float, str, Page]] = []
    for category, groups in strata.items():
        spread = []
        for name, members in groups.items():
            # A random offset per group, so that pages of groups too small to appear in
            # every sample are drawn with the same probability as any other page.
            offset = _draw(seed, category, *name)
            members.sort(key=lambda p: (_draw(seed, p.key), p.key))
            spread += [((i + offset) / len(members), p.key, p) for i, p in enumerate(members)]
        spread.sort(key=lambda item: item[:2])
        ranked += [((i + 0.5) / len(spread), p.key, p) for i, (_, _, p) in enumerate(spread)]
    ranked.sort(key=lambda item: item[:2])
    return [page for _, _, page in ranked[:total]]


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


def _stream_members(kind: str):
    """(stem, bytes reader) for each training file, streamed instead of downloaded."""
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/train/text/{kind}.tar.gz"
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith(SUFFIX[kind]):
                name = member.name.replace("\\", "/").rsplit("/", 1)[-1]
                yield name[: -len(SUFFIX[kind])], lambda m=member: tar.extractfile(m).read()


def train_pages(root: Path, count: int) -> list[Page]:
    """`count` training pages with ground truth, spread evenly over the training archive."""
    folder = root / f"train_pilot_{count}"
    if not (folder / "done").exists():
        wanted: set[str] = set()
        (folder / "train" / "img").mkdir(parents=True, exist_ok=True)
        (folder / "train" / "gt").mkdir(parents=True, exist_ok=True)
        step = TRAIN_TEXT_PAGES // count
        for i, (stem, read) in enumerate(_stream_members("img")):
            if i % step == 0:
                (folder / "train" / "img" / f"{stem}.jpg").write_bytes(read())
                wanted.add(stem)
                if len(wanted) == count:
                    break
        for stem, read in _stream_members("gt"):
            if stem in wanted:
                (folder / "train" / "gt" / f"{stem}.txt").write_bytes(read())
                wanted.remove(stem)
                if not wanted:
                    break
        (folder / "done").touch()
    return collect_pages(folder, ("train",))


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
