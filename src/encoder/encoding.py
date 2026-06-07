import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.encoder.dinov3_encoder import DINOv3Encoder
from src.utils.boxes import crops_from_boxes


def encode_detections(
    encoder: DINOv3Encoder,
    detections,
    out_dir,
    batch_size=32,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    records_meta = []

    for img_idx, entry in enumerate(detections):
        img_path = entry["image_path"]
        boxes = entry.get("boxes_xyxy", [])
        scores = entry.get("scores", [])
        gt_categories = entry.get("gt_categories")

        if not boxes:
            continue

        with Image.open(img_path) as img:
            image_rgb = np.asarray(img.convert("RGB"))
        crops = crops_from_boxes(image_rgb, boxes)

        if not crops:
            continue

        features = encoder.encode_batch(crops, batch_size=batch_size)

        for i, box in enumerate(boxes):
            record = {
                "image_path": img_path,
                "box_xyxy": box,
                "score": scores[i] if i < len(scores) else 0.0,
                "feature_index": len(all_features),
            }
            if gt_categories is not None and i < len(gt_categories):
                record["gt_category"] = gt_categories[i]

            records_meta.append(record)
            all_features.append(features[i])

        if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(detections):
            print(f"[encode] {img_idx + 1}/{len(detections)} | crops={len(all_features)}")

    if all_features:
        features_array = np.stack(all_features).astype(np.float32)
    else:
        embed_dim = int(getattr(encoder, "embed_dim", 0))
        features_array = np.zeros((0, embed_dim), dtype=np.float32)

    np.save(out_dir / "features.npy", features_array)
    (out_dir / "records.json").write_text(json.dumps(records_meta, indent=2))
    print(f"[encode] Saved {len(all_features)} features -> {out_dir}")

    return features_array, records_meta