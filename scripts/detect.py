from scripts.common import collect_images, resolve_path
from src.detectors import DDETRDetector

from src.detectors import detect_all_images
from src.utils import release_cuda_cache


def stage_detect(cfg, out_path, log_fn=print):
    log_fn("\n[detect] collecting images")
    images = collect_images(resolve_path(cfg["images_dir"]))
    log_fn(f"[detect] found {len(images)} images")

    detector = DDETRDetector(
        repo_root=resolve_path(cfg["det_repo"]),
        checkpoint_path=resolve_path(cfg["det_ckpt"]),
        device=cfg.get("device", "cuda"),
    )
    detect_all_images(
        detector,
        images,
        out_path,
        score_thresh=cfg.get("score_thresh", 0.4),
        topk=cfg.get("topk", 100),
        min_box_side=cfg.get("min_box_side", 2.0),
    )
    del detector
    release_cuda_cache()
    log_fn(f"[detect] saved -> {out_path}")
