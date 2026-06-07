import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.common import resolve_path
from src.evaluation.coco_utils import image_id_from_path


def load_features(features_dir: Path):
    features = np.load(features_dir / "features.npy")
    records = json.loads((features_dir / "records.json").read_text())
    return features, records


def load_support_set_from_memory_meta(memory_path: str | Path) -> set[int]:
    # Support feature indices saved at memory build (for open-set calibration exclusion)
    path = Path(memory_path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.is_file():
        return set()
    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    raw = data.get("support_feature_indices", [])
    if not isinstance(raw, list):
        return set()
    return {int(x) for x in raw}


def load_split_image_ids(split_file: str | Path):
    split_path = resolve_path(split_file)
    ids = set()
    for line in split_path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ids.add(image_id_from_path(s))
        except ValueError:
            continue
    return ids


def filter_records_by_split(records, split_file, log_fn=None):
    if not split_file:
        return records
    ids = load_split_image_ids(split_file)
    filtered = []
    for r in records:
        try:
            img_id = image_id_from_path(r["image_path"])
        except (KeyError, ValueError):
            continue
        if img_id in ids:
            filtered.append(r)
    if log_fn is not None:
        log_fn(f"[split] {len(filtered)}/{len(records)} records kept from split: {split_file}")
    return filtered


def build_class_crops(records):
    class_crops = defaultdict(list)
    for record in records:
        gt_category = record.get("gt_category")
        if gt_category:
            class_crops[gt_category].append(record)
    return class_crops

