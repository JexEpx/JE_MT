from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.coco_utils import image_id_from_path, load_coco_gt, match_predictions_to_gt
from src.evaluation.owod_split import COCO_CLASSES, OWODSplit
from src.utils.boxes import clip_boxes_xyxy, valid_box_mask


def _gt_box_score(box_xyxy, image_h, image_w):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    img_area = max(1.0, float(image_h) * float(image_w))
    return float((box_w * box_h) / img_area)


def _load_gt_by_image(ann_paths, image_ids):
    if isinstance(ann_paths, (str, Path)):
        ann_paths = [ann_paths]
    ann_paths = [Path(p) for p in ann_paths]

    all_ids = set(COCO_CLASSES.keys())
    dummy_split = OWODSplit(task=0, known_ids=list(all_ids), unknown_ids=[])

    gt_by_image = {}
    for ann_path in ann_paths:
        partial = load_coco_gt(ann_path, dummy_split, image_ids=image_ids)
        for img_id, boxes in partial.items():
            gt_by_image.setdefault(img_id, []).extend(boxes)

    return gt_by_image, ann_paths


def build_gt_from_images(image_paths, ann_paths, min_box_side=2.0):
    image_ids = set()
    for path in image_paths:
        try:
            image_ids.add(image_id_from_path(str(path)))
        except ValueError:
            continue

    gt_by_image, ann_paths = _load_gt_by_image(ann_paths, image_ids)

    results = []
    for img_path in image_paths:
        img_path = Path(img_path)
        try:
            img_id = image_id_from_path(str(img_path))
        except ValueError:
            results.append({"image_path": str(img_path.resolve()), "boxes_xyxy": [], "scores": [], "gt_categories": []})
            continue

        gts = gt_by_image.get(img_id, [])
        if not gts:
            results.append({"image_path": str(img_path.resolve()), "boxes_xyxy": [], "scores": [], "gt_categories": []})
            continue

        image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        h, w = image_rgb.shape[:2]
        boxes_list, cats_list, scores_list = [], [], []
        for gt in gts:
            box = clip_boxes_xyxy(gt.bbox_xyxy[np.newaxis, :], h, w)[0]
            if not valid_box_mask(box[np.newaxis, :], min_side=min_box_side)[0]:
                continue
            boxes_list.append([float(v) for v in box])
            cats_list.append(gt.category_name)
            scores_list.append(_gt_box_score(box, h, w))

        results.append({
            "image_path": str(img_path.resolve()),
            "boxes_xyxy": boxes_list,
            "scores": scores_list,
            "gt_categories": cats_list,
        })

    total = sum(len(r["boxes_xyxy"]) for r in results)
    ann_names = ", ".join(p.name for p in ann_paths)
    print(f"[gt_boxes] COCO GT: {total} boxes across {len(results)} images -> {ann_names}")
    return results


def attach_gt_to_detections(detections, ann_paths, iou_threshold=0.5):
    image_ids = set()
    for entry in detections:
        try:
            image_ids.add(image_id_from_path(str(entry["image_path"])))
        except (KeyError, ValueError):
            continue

    gt_by_image, ann_paths = _load_gt_by_image(ann_paths, image_ids)
    ann_names = ", ".join(p.name for p in ann_paths)

    matched = 0
    total = 0
    for entry in detections:
        boxes = entry.get("boxes_xyxy", [])
        gt_categories = [None] * len(boxes)
        total += len(boxes)
        try:
            img_id = image_id_from_path(str(entry["image_path"]))
        except (KeyError, ValueError):
            entry["gt_categories"] = gt_categories
            continue

        gts = gt_by_image.get(img_id, [])
        if not gts or not boxes:
            entry["gt_categories"] = gt_categories
            continue

        pred_boxes = np.asarray(boxes, dtype=np.float32)
        pairs = match_predictions_to_gt(pred_boxes=pred_boxes, gt_boxes=gts, iou_threshold=float(iou_threshold))
        for pi, gi in pairs:
            gt_categories[pi] = gts[gi].category_name
            matched += 1
        entry["gt_categories"] = gt_categories

    print(f"[gt_match] matched {matched}/{total} detections with GT (IoU>={iou_threshold}) from {ann_names}")
    return detections
