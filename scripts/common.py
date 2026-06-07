import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def collect_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)


def subsample_images(images: list, k: int, seed: int = 42, shuffle: bool = False) -> list:
    if k <= 0 or k >= len(images):
        return images

    if shuffle:
        return random.Random(seed).sample(images, k)

    return images[:k]
