from pathlib import Path


def parse_image_id_flexible(value: str | Path) -> int:
    # Parse image ID from an integer, path, or COCO-like filename
    s = str(value).strip()
    if not s:
        raise ValueError("empty image id/path")

    if s.isdigit():
        return int(s)

    stem = Path(s).stem.split("__", 1)[0]
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break

    if not digits:
        raise ValueError(f"Cannot parse image id from: {value}")
    return int(digits)
