from scripts.common import collect_images, resolve_path
from src.detectors import load_detections
from src.encoder import DINOv3Encoder, encode_detections
from src.evaluation import attach_gt_to_detections, build_gt_from_images
from src.utils import release_cuda_cache


def is_coco_annotation_json(data) -> bool:
    return isinstance(data, dict) and "images" in data and "annotations" in data


def is_detection_list(data) -> bool:
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return isinstance(first, dict) and "image_path" in first and "boxes_xyxy" in first


def stage_encode(cfg, out_dir, log_fn=print):
    det_paths = [resolve_path(p) for p in cfg["det_paths"]]
    ann_paths = [resolve_path(p) for p in cfg.get("ann_paths", [])]
    images = collect_images(resolve_path(cfg["images_dir"]))

    log_fn("\n[encode] loading detections")
    loaded_list = [load_detections(p) for p in det_paths]

    if all(is_detection_list(x) for x in loaded_list):
        detections = []
        for item in loaded_list:
            detections.extend(item)
        if ann_paths:
            detections = attach_gt_to_detections(
                detections=detections,
                ann_paths=ann_paths,
                iou_threshold=float(cfg.get("gt_assign_iou_threshold", 0.5)),
            )
    elif all(is_coco_annotation_json(x) for x in loaded_list):
        log_fn("[encode] COCO annotations detected; building GT boxes per image")
        src_ann_paths = ann_paths if ann_paths else det_paths
        detections = build_gt_from_images(
            image_paths=images,
            ann_paths=src_ann_paths,
            min_box_side=float(cfg.get("min_box_side", 2.0)),
        )
    else:
        joined = ", ".join(str(p) for p in det_paths)
        raise ValueError(
            # Mixed/unsupported detections formats across files
            f"{joined}. Use all detection-list JSONs or all COCO instances JSONs."
        )

    batch_size = int(cfg.get("batch_size", 8))
    encoder = DINOv3Encoder(
        repo_or_dir=resolve_path(cfg["enc_repo"]),
        model_name=cfg.get("enc_model", "dinov3_vitb16"),
        weights=resolve_path(cfg["enc_weights"]),
        device=cfg.get("enc_device", "cuda"),
        image_size=cfg.get("enc_image_size", 224),
    )

    features, records = encode_detections(
        encoder=encoder,
        detections=detections,
        out_dir=out_dir,
        batch_size=batch_size,
    )
    del encoder
    release_cuda_cache()

    log_fn(f"[encode] features={features.shape} records={len(records)}")
    return out_dir

