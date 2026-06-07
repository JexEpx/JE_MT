import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.boxes import clip_boxes_xyxy, valid_box_mask


def detect_all_images(
    detector,
    image_paths,
    out_path,
    score_thresh=0.4,
    min_box_side=2.0,
    topk=100,
):
    results = []
    total_boxes = 0

    for idx, img_path in enumerate(image_paths):
        with Image.open(img_path) as img:
            image_rgb = np.asarray(img.convert("RGB"))

        h, w = image_rgb.shape[:2]

        boxes, scores = detector.predict(image_rgb)
        boxes = clip_boxes_xyxy(boxes, h, w)

        keep = valid_box_mask(boxes, min_side=min_box_side) & (scores >= score_thresh)
        indices = np.where(keep)[0]
        if topk > 0 and len(indices) > topk:
            indices = indices[np.argsort(scores[indices])[::-1][:topk]]

        total_boxes += len(indices)

        results.append({
            "image_path": str(img_path),
            "boxes_xyxy": boxes[indices].tolist(),
            "scores": scores[indices].tolist(),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"[detect] {idx + 1}/{len(image_paths)} | boxes={total_boxes}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"[detect] Saved {total_boxes} boxes across {len(results)} images -> {out_path}")
    return out_path


def load_detections(det_path):
    return json.loads(Path(det_path).read_text())
