import json

import torch

from src.evaluation import OWODMetrics, get_split, resolve_split_task
from src.memory import UNKNOWN, PrototypeMemory
from src.memory.open_set_defaults import apply_open_set_defaults

from scripts.common import resolve_path
from scripts.utils import (
    filter_records_by_split,
    load_features,
    load_support_set_from_memory_meta,
)


def _refit_open_set_gates_if_needed(memory, cfg, memory_path, log_fn=print) -> None:
    # Re-calibrate gates when eval-time MHN settings differ from threshold fitting
    if not (
        memory.use_global_threshold
        or memory.use_proto_margin
        or memory._uses_maha_gate()
    ):
        return
    calib_refine = bool(getattr(memory, "open_set_calib_use_mhn_refine", False))
    calib_classify = bool(getattr(memory, "open_set_calib_use_mhn_classify", False))
    gate_cosine = bool(getattr(memory, "open_set_gate_use_cosine_class", True))
    refine_mismatch = memory.use_mhn_refine != calib_refine
    classify_mismatch = (
        memory.use_mhn_classify != calib_classify and not gate_cosine
    )
    if not refine_mismatch and not classify_mismatch:
        return
    calib_split = cfg.get("open_set_calib_split_file") or cfg.get("train_split_file")
    if not calib_split:
        log_fn(
            "[evaluate] WARNING: MHN eval settings differ from calibration but no "
            "open_set_calib_split_file/train_split_file in cfg; gates may be misaligned"
        )
        return
    log_fn(
        f"[evaluate] re-fitting open-set gates: calib refine={calib_refine} "
        f"classify={calib_classify} -> infer refine={memory.use_mhn_refine} "
        f"classify={memory.use_mhn_classify}"
    )
    features_dir = resolve_path(cfg["features_dir"])
    features, cal_records = load_features(features_dir)
    cal_records = filter_records_by_split(cal_records, calib_split, log_fn=log_fn)
    support_set = load_support_set_from_memory_meta(memory_path)
    split_task = resolve_split_task(cfg)
    known_names = (
        get_split(split_task).known_names if split_task is not None else list(memory.labels)
    )
    calib_cfg = apply_open_set_defaults(dict(cfg))
    max_records = calib_cfg.get("open_set_calib_max_records")
    memory.open_set_calib_max_records = None if max_records in (None, 0) else int(max_records)
    memory.open_set_calib_seed = int(calib_cfg.get("open_set_calib_seed", 0))
    memory.fit_open_set_thresholds(
        features,
        cal_records,
        known_names,
        method=str(calib_cfg.get("open_set_calib_method", "known_only")),
        known_percentile=float(calib_cfg.get("open_set_calib_known_percentile", 5.0)),
        known_margin_percentile=float(
            calib_cfg.get("open_set_calib_known_margin_percentile", 5.0)
        ),
        support_set=support_set or None,
        log_fn=log_fn,
    )


def _apply_eval_cfg(memory, cfg) -> None:
    for key in ("use_global_threshold", "use_proto_margin"):
        if key in cfg:
            setattr(memory, key, bool(cfg[key]))
    if "use_mhn_classify" in cfg:
        memory.use_mhn_classify = bool(cfg["use_mhn_classify"])
    if "use_mhn_refine" in cfg:
        memory.use_mhn_refine = bool(cfg["use_mhn_refine"])
    if "mhn_beta" in cfg:
        memory.mhn.beta = float(cfg["mhn_beta"])
    if "open_set_margin_mode" in cfg:
        mode = str(cfg["open_set_margin_mode"]).strip().lower()
        if mode not in {"class_aware", "global_proto"}:
            raise ValueError(f"open_set_margin_mode must be class_aware|global_proto, got {mode!r}")
        memory.open_set_margin_mode = mode
    if "open_set_gate_use_cosine_class" in cfg:
        memory.open_set_gate_use_cosine_class = bool(cfg["open_set_gate_use_cosine_class"])


def stage_evaluate(cfg, results_dir, support_set=None, log_fn=print):
    features_dir = resolve_path(cfg["features_dir"])
    memory_path = resolve_path(cfg["memory_path"])

    log_fn("\n[evaluate] loading test features")
    features, records = load_features(features_dir)
    records = filter_records_by_split(records, cfg.get("test_split_file", ""), log_fn=log_fn)

    memory = PrototypeMemory.load(memory_path, device=cfg.get("device", "cpu"))
    _apply_eval_cfg(memory, cfg)
    _refit_open_set_gates_if_needed(memory, cfg, memory_path, log_fn=log_fn)

    tau_g = (
        float(memory.global_top1_threshold.item())
        if memory.global_top1_threshold is not None
        else None
    )
    tau_m = (
        float(memory.margin_threshold.item())
        if memory.margin_threshold is not None
        else None
    )
    calib_refine = bool(getattr(memory, "open_set_calib_use_mhn_refine", False))
    calib_classify = bool(getattr(memory, "open_set_calib_use_mhn_classify", False))
    gate_cosine = bool(getattr(memory, "open_set_gate_use_cosine_class", True))
    log_fn(
        f"[evaluate] inference: cls={memory.use_mhn_classify} refine={memory.use_mhn_refine} "
        f"gate_cosine_class={gate_cosine} calib_classify={calib_classify} calib_refine={calib_refine} "
        f"open_set_mode={memory.open_set_threshold_mode} margin_mode={memory.open_set_margin_mode} "
        f"tau_top1={tau_g} tau_margin={tau_m} "
        f"global_gate={memory.use_global_threshold} margin_gate={memory.use_proto_margin}"
    )

    split_task = resolve_split_task(cfg)
    known_names = get_split(split_task).known_names if split_task is not None else list(memory.labels)
    known_set = set(known_names)
    label_to_idx = {name: i for i, name in enumerate(memory.labels)}
    metrics = OWODMetrics(known_names)

    predictions = []
    correct = total = unknown_preds = 0
    for r in records:
        idx = int(r["feature_index"])
        if support_set and idx in support_set:
            continue
        true = r.get("gt_category")
        if not true:
            continue

        z = torch.from_numpy(features[idx]).float().to(memory.device)
        z = memory.prepare_for_inference(z)
        proto_sims = z @ memory.prototypes.T if memory.prototypes.numel() > 0 else None
        pred, conf = memory.predict(z, prepared=True)

        open_set_top1 = open_set_margin = 0.0
        if memory.num_classes > 0:
            gate_cidx = memory._class_index_for_open_set_gate(z)
            gate_top1, gate_margin = memory.open_set_signals(z, gate_cidx)
            open_set_top1 = float(gate_top1.item())
            open_set_margin = float(gate_margin.item())

        true_bucket = true if true in known_set else UNKNOWN
        if (
            true_bucket != UNKNOWN
            and true_bucket in label_to_idx
            and proto_sims is not None
        ):
            true_class_idx = label_to_idx[true_bucket]
            class_mask = memory.proto_class == true_class_idx
            if class_mask.any():
                true_class_sims = proto_sims[class_mask]
                true_class_nearest_proto_cosine = float(true_class_sims.max().item())
                true_class_proto_distance = float(1.0 - true_class_nearest_proto_cosine)
            else:
                true_class_nearest_proto_cosine = None
                true_class_proto_distance = None
        else:
            true_class_nearest_proto_cosine = None
            true_class_proto_distance = None

        total += 1
        if pred == true_bucket:
            correct += 1
        if pred == UNKNOWN:
            unknown_preds += 1
        metrics.add(pred, true)
        predictions.append(
            {
                "image_path": r.get("image_path"),
                "box_xyxy": r.get("box_xyxy"),
                "feature_index": idx,
                "score": float(r.get("score", 0.0)),
                "true_label": true,
                "true_bucket": true_bucket,
                "pred_label": pred,
                "pred_conf": float(conf),
                "open_set_top1": open_set_top1,
                "open_set_margin": open_set_margin,
                "true_class_nearest_proto_cosine": true_class_nearest_proto_cosine,
                "true_class_proto_distance": true_class_proto_distance,
            }
        )

    acc = 100.0 * correct / max(1, total)
    result = {
        "accuracy": acc,
        "unknown_predictions": unknown_preds,
        "total": total,
        "memory_path": str(memory_path),
    }
    owod = metrics.compute()
    metrics.print_report(owod)
    owod_dict = owod.to_dict()
    log_fn(f"[evaluate] acc={acc:.2f}% unknown_preds={unknown_preds}/{total}")

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "owod_eval.json").write_text(json.dumps(owod_dict, indent=2))
    log_fn(f"[evaluate] OWOD metrics -> {results_dir / 'owod_eval.json'}")

    combined = {
        "result": result,
        "owod_feature_metrics": owod_dict,
        "predictions": predictions,
    }
    (results_dir / "combined_results.json").write_text(json.dumps(combined, indent=2))
    log_fn(f"[evaluate] combined results -> {results_dir / 'combined_results.json'}")
