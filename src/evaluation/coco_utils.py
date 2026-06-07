import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.owod_split import COCO_CLASSES, OWODSplit
from src.utils.image_ids import parse_image_id_flexible


def image_id_from_path(image_path: str) -> int:
    # Extract image ID from the image path
    return parse_image_id_flexible(image_path)


@dataclass
class GTBox:

    image_id: int
    category_id: int
    category_name: str
    bbox_xyxy: np.ndarray
    is_known: bool
    is_crowd: bool = False


def _xywh_to_xyxy(bbox: list[float]) -> np.ndarray:
    # Convert a COCO bbox from [x, y, w, h] to [x1, y1, x2, y2]
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)



def load_coco_gt(
    ann_path: str | Path,
    split: OWODSplit,
    image_ids: set[int] | None = None,
    skip_crowd: bool = True,
    min_area: float = 0.0,
) -> dict[int, list[GTBox]]:

    raw = json.loads(Path(ann_path).read_text())
    valid_cat_ids = set(COCO_CLASSES.keys())

    gt: dict[int, list[GTBox]] = {}

    for ann in raw["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in valid_cat_ids:
            continue
        if skip_crowd and ann.get("iscrowd", 0):
            continue
        if ann.get("area", 1) < min_area:
            continue

        img_id = ann["image_id"]
        if image_ids is not None and img_id not in image_ids:
            continue

        gt.setdefault(img_id, []).append(
            GTBox(
                image_id=img_id,
                category_id=cat_id,
                category_name=COCO_CLASSES[cat_id],
                bbox_xyxy=_xywh_to_xyxy(ann["bbox"]),
                is_known=split.is_known(cat_id),
                is_crowd=bool(ann.get("iscrowd", 0)),
            )
        )

    return gt



def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    # Compute pairwise IoU matrix for boxes_a (M,4) and boxes_b (N,4)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    gt_boxes: list[GTBox],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    # Greedy highest-IoU matching returning (pred_idx, gt_idx) pairs
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    gt_xyxy = np.stack([g.bbox_xyxy for g in gt_boxes], axis=0)
    iou = compute_iou_matrix(pred_boxes, gt_xyxy)

    matches: list[tuple[int, int]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Highest IoU first
    while True:
        remaining = iou.copy()
        remaining[list(matched_pred), :] = 0
        remaining[:, list(matched_gt)] = 0

        best = remaining.max()
        if best < iou_threshold:
            break

        pi, gi = np.unravel_index(remaining.argmax(), remaining.shape)
        matches.append((int(pi), int(gi)))
        matched_pred.add(int(pi))
        matched_gt.add(int(gi))

    return matches


