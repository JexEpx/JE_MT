import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.common import resolve_path
from src.memory.memory_builder import build_memory, select_class_feature_indices
from src.evaluation import get_split, resolve_split_task
from src.memory import PrototypeMemory

from scripts.memory_build_progress import BuildProgressTracker, save_build_progress
from scripts.utils import (
    build_class_crops,
    filter_records_by_split,
    load_features,
)
from src.memory.open_set_defaults import OPEN_SET_GATE_DEFAULTS

MEM_CFG_KEYS = (
    "device",
    "random_seed",
    "use_global_threshold",
    "use_proto_margin",
    "open_set_calib_known_percentile",
    "open_set_calib_known_margin_percentile",
    "open_set_calib_method",
    "open_set_margin_mode",
    "open_set_gate_mode",
    "open_set_maha_mode",
    "open_set_maha_distance",
    "open_set_maha_eps",
    "open_set_calib_maha_percentile",
    "maha_threshold",
    "use_mhn_classify",
    "use_mhn_refine",
    "mhn_beta",
    "exemplar_mode",
    "num_prototypes",
    "prototype_init",
    "exemplar_max_per_class",
    "support_min_cosine",
)


def _mem_cfg_from_fit(cfg):
    return {k: cfg[k] for k in MEM_CFG_KEYS if k in cfg}


def _apply_memory_runtime_cfg(memory, cfg) -> None:
    # Apply fit/eval flags (MHN, gates, margins) from cfg onto a live memory bank
    for key in ("use_global_threshold", "use_proto_margin"):
        if key in cfg:
            setattr(memory, key, bool(cfg[key]))
    if "use_mhn_classify" in cfg:
        memory.use_mhn_classify = bool(cfg["use_mhn_classify"])
    if "use_mhn_refine" in cfg:
        memory.use_mhn_refine = bool(cfg["use_mhn_refine"])
    if "mhn_beta" in cfg:
        memory.mhn.beta = float(cfg["mhn_beta"])
    if "open_set_gate_use_cosine_class" in cfg:
        memory.open_set_gate_use_cosine_class = bool(cfg["open_set_gate_use_cosine_class"])
    if "open_set_margin_mode" in cfg:
        mode = str(cfg["open_set_margin_mode"]).strip().lower()
        if mode not in {"class_aware", "global_proto"}:
            raise ValueError(f"open_set_margin_mode must be class_aware|global_proto, got {mode!r}")
        memory.open_set_margin_mode = mode


def _calibrate_open_set_gates(
    memory,
    features_dir,
    cfg,
    split_task,
    support_set,
    *,
    features=None,
    records=None,
    log_fn=print,
) -> dict:
    calib_split = (
        cfg.get("open_set_calib_split_file")
        or cfg.get("train_split_file")
        or cfg.get("test_split_file", "")
    )
    if not calib_split:
        log_fn("[open_set_calib] skipped: set open_set_calib_split_file or train_split_file")
        return {}

    if not (memory.use_global_threshold or memory.use_proto_margin or memory._uses_maha_gate()):
        return {}

    _apply_memory_runtime_cfg(memory, cfg)

    train_split = cfg.get("train_split_file", "")
    need_full_calib_pool = bool(
        calib_split
        and train_split
        and str(calib_split) != str(train_split)
    )
    if features is None or records is None or need_full_calib_pool:
        features, cal_records = load_features(features_dir)
        cal_records = filter_records_by_split(cal_records, calib_split, log_fn=log_fn)
    else:
        cal_records = records
    if split_task is None:
        known_names = list(memory.labels)
    else:
        known_names = get_split(split_task).known_names

    method = str(cfg.get("open_set_calib_method", "known_only"))
    memory.open_set_calib_known_percentile = float(
        cfg.get("open_set_calib_known_percentile", 5.0)
    )
    memory.open_set_calib_known_margin_percentile = float(
        cfg.get("open_set_calib_known_margin_percentile", 5.0)
    )
    max_records = cfg.get(
        "open_set_calib_max_records", OPEN_SET_GATE_DEFAULTS["open_set_calib_max_records"]
    )
    memory.open_set_calib_max_records = None if max_records in (None, 0) else int(max_records)
    memory.open_set_calib_seed = int(
        cfg.get("open_set_calib_seed", OPEN_SET_GATE_DEFAULTS["open_set_calib_seed"])
    )
    return memory.fit_open_set_thresholds(
        features,
        cal_records,
        known_names,
        method=method,
        known_percentile=memory.open_set_calib_known_percentile,
        known_margin_percentile=memory.open_set_calib_known_margin_percentile,
        support_set=support_set,
        log_fn=log_fn,
    )


def _class_prototype_mean(memory, class_idx):
    mask = memory.proto_class == class_idx
    if not bool(mask.any()):
        return None
    protos = memory.prototypes[mask]
    return F.normalize(protos.mean(dim=0), dim=0)


def _passes_online_min_cosine(memory, z, class_idx, min_cosine):
    min_cos = float(min_cosine)
    if min_cos <= 0.0:
        return True
    ref = _class_prototype_mean(memory, class_idx)
    if ref is None:
        return True
    z = F.normalize(z.float().to(ref.device), dim=-1)
    return float(torch.dot(z, ref)) >= min_cos


def _proportional_class_quotas(class_sizes: dict[str, int], budget: int) -> dict[str, int]:
    # Split budget across classes by pool size
    if budget <= 0 or not class_sizes:
        return {cls: 0 for cls in class_sizes}
    total = sum(class_sizes.values())
    if total <= 0:
        return {cls: 0 for cls in class_sizes}

    raw = {cls: budget * n / total for cls, n in class_sizes.items()}
    quotas = {cls: int(raw[cls]) for cls in class_sizes}
    shortfall = budget - sum(quotas.values())
    if shortfall > 0:
        order = sorted(class_sizes, key=lambda c: raw[c] - quotas[c], reverse=True)
        for i in range(shortfall):
            quotas[order[i % len(order)]] += 1
    return quotas


def _prepare_online_records(
    records,
    support_set,
    label_to_idx,
    *,
    online_max=0,
    random_seed=153,
    log_fn=None,
):
    # Known-class crops only; seeded shuffle; proportional cap when online_max > 0
    pool = [
        r for r in records
        if int(r["feature_index"]) not in support_set
        and r.get("gt_category") in label_to_idx
    ]
    rng = random.Random(int(random_seed))

    if online_max <= 0:
        rng.shuffle(pool)
        if log_fn is not None:
            log_fn(f"online pool: {len(pool)} known-class crops (full, shuffled)")
        return pool

    by_class: dict[str, list] = {name: [] for name in label_to_idx}
    for r in pool:
        by_class[r["gt_category"]].append(r)

    sizes = {cls: len(crops) for cls, crops in by_class.items() if crops}
    pool_total = sum(sizes.values())
    if pool_total <= online_max:
        chosen = list(pool)
        rng.shuffle(chosen)
        if log_fn is not None:
            log_fn(f"online pool: {len(chosen)} known-class crops (all available, shuffled)")
        return chosen

    quotas = _proportional_class_quotas(sizes, online_max)
    chosen: list = []
    used: set[int] = set()
    for cls in sorted(by_class):
        cls_pool = by_class[cls]
        rng.shuffle(cls_pool)
        take = min(quotas.get(cls, 0), len(cls_pool))
        picked = cls_pool[:take]
        chosen.extend(picked)
        used.update(int(r["feature_index"]) for r in picked)

    if len(chosen) < online_max:
        remainder = [
            r for r in pool if int(r["feature_index"]) not in used
        ]
        rng.shuffle(remainder)
        chosen.extend(remainder[: online_max - len(chosen)])

    rng.shuffle(chosen)
    chosen = chosen[:online_max]

    if log_fn is not None:
        log_fn(
            f"online pool: {len(chosen)}/{pool_total} known-class crops "
            f"(proportional cap, seed={random_seed})"
        )
    return chosen


def _online_refine_memory(
    memory,
    features,
    records,
    support_set,
    *,
    tau_update,
    tau_new,
    alpha,
    online_max=0,
    online_min_cosine=0.15,
    random_seed=153,
    progress_tracker: BuildProgressTracker | None = None,
    log_fn=print,
):
    label_to_idx = {name: i for i, name in enumerate(memory.labels)}
    train_records = _prepare_online_records(
        records,
        support_set,
        label_to_idx,
        online_max=online_max,
        random_seed=random_seed,
        log_fn=log_fn,
    )

    min_cos = float(online_min_cosine)
    updates = 0
    skipped = 0
    total = len(train_records)
    for i, r in enumerate(train_records, start=1):
        gt = r["gt_category"]
        cidx = label_to_idx[gt]
        z = torch.from_numpy(features[int(r["feature_index"])]).float().to(memory.device)
        z = F.normalize(z, dim=-1)

        if not _passes_online_min_cosine(memory, z, cidx, min_cos):
            skipped += 1
            continue

        if memory.update(z, cidx, tau_update=tau_update, tau_new=tau_new, alpha=alpha):
            updates += 1
            if progress_tracker is not None:
                progress_tracker.on_online_step(i, updates)

    if progress_tracker is not None:
        progress_tracker.snapshot_finish(total, updates)

    log_fn(
        f"online refinement: {updates}/{total} updates"
        + (f" ({skipped} skipped by online_min_cosine)" if skipped else "")
    )
    return updates


def stage_fit_memory(out_path, cfg, memory_in=None, log_fn=print):
    features_dir = resolve_path(cfg["features_dir"])
    train_split_file = cfg.get("train_split_file", "")
    split_task = resolve_split_task(cfg)
    n_support = int(cfg.get("n_support", 5))
    online_refine = bool(cfg.get("online_refine", False))
    online_max = int(cfg.get("online_max", 0))
    tau_update = float(cfg.get("tau_update", 0.7))
    tau_new = float(cfg.get("tau_new", 0.5))
    alpha = float(cfg.get("alpha", 0.1))
    online_min_cosine = float(cfg.get("online_min_cosine", 0.15))
    num_prototypes = max(1, int(cfg.get("num_prototypes", 1)))
    prototype_init = str(cfg.get("prototype_init", "kmeans"))
    exemplar_mode = bool(cfg.get("exemplar_mode", False))
    exemplar_max_per_class = int(cfg.get("exemplar_max_per_class", 0))
    random_seed = int(cfg.get("random_seed", 153))

    memory_in_path = Path(memory_in) if memory_in else None
    resume = memory_in_path is not None and memory_in_path.is_file()

    log_fn(
        f"\n[fit_memory] loading features"
        + (f" (resume <- {memory_in_path})" if resume else " (fresh build)")
    )
    features, records = load_features(features_dir)
    records = filter_records_by_split(records, train_split_file, log_fn=log_fn)

    mem_cfg = _mem_cfg_from_fit(cfg)

    class_crops = build_class_crops(records)
    if split_task is not None:
        class_names = [n for n in get_split(split_task).known_names if n in class_crops]
    else:
        class_names = sorted(class_crops.keys())

    support_set = set()
    new_classes = 0
    support_updates = 0

    if resume:
        memory = PrototypeMemory.load(memory_in_path, device=cfg.get("device", "cpu"))
        memory.update_mode = "append_only" if exemplar_mode else "ema"
        label_to_idx = {name: i for i, name in enumerate(memory.labels)}

        for cls in class_names:
            crops = class_crops.get(cls, [])
            idxs = select_class_feature_indices(
                crops,
                n_support=n_support,
                exemplar_mode=exemplar_mode,
                exemplar_max_per_class=exemplar_max_per_class,
                features=features,
                random_seed=random_seed,
                support_min_cosine=float(mem_cfg.get("support_min_cosine", 0.15)),
                log_fn=log_fn,
            )
            if not idxs:
                continue
            support_set.update(idxs)

            if cls in label_to_idx:
                cidx = label_to_idx[cls]
                for idx in idxs:
                    z = torch.from_numpy(features[idx]).float().to(memory.device)
                    if memory.update(z, cidx, tau_update=tau_update, tau_new=tau_new, alpha=alpha):
                        support_updates += 1
            else:
                memory.add_class(
                    torch.from_numpy(features[idxs]).float(),
                    label=cls,
                    num_prototypes=(len(idxs) if exemplar_mode else num_prototypes),
                    prototype_init=("examples" if exemplar_mode else prototype_init),
                )
                label_to_idx[cls] = len(memory.labels) - 1
                new_classes += 1

    else:
        memory, support_set = build_memory(
            features,
            class_crops,
            class_names,
            n_support,
            mem_cfg,
            log_fn=log_fn,
        )

    _apply_memory_runtime_cfg(memory, cfg)

    progress_every = int(cfg.get("progress_eval_every_updates", 0))
    if not online_refine:
        progress_every = 0

    progress_tracker = None
    if progress_every > 0:
        test_split = cfg.get("test_split_file", "")
        if not test_split:
            log_fn("[build_progress] skipped: no test_split_file in cfg")
            progress_every = 0
        else:
            log_fn(f"[build_progress] eval every {progress_every} online updates")
            _, test_records = load_features(features_dir)
            test_records = filter_records_by_split(test_records, test_split, log_fn=log_fn)
            progress_eval_cfg = {
                "features_dir": str(features_dir),
                "test_split_file": test_split,
                "split_task": split_task if split_task is not None else cfg.get("split_task"),
                "device": cfg.get("device", "cpu"),
                "use_global_threshold": bool(cfg.get("use_global_threshold", True)),
                "use_proto_margin": bool(cfg.get("use_proto_margin", True)),
                "use_mhn_classify": False,
                "use_mhn_refine": False,
                "mhn_beta": float(cfg.get("mhn_beta", 30.0)),
            }
            progress_calib = bool(cfg.get("progress_calib_open_set", True))

            def _progress_calibrate_gates() -> None:
                _calibrate_open_set_gates(
                    memory,
                    features_dir,
                    cfg,
                    split_task,
                    support_set,
                    features=features,
                    records=records,
                    log_fn=lambda msg: log_fn(f"[build_progress] {msg}"),
                )

            progress_tracker = BuildProgressTracker(
                memory,
                features,
                test_records,
                progress_eval_cfg,
                support_set,
                eval_every_updates=progress_every,
                log_fn=log_fn,
                calibrate_gates=_progress_calibrate_gates if progress_calib else None,
            )
            progress_tracker.snapshot_post_build()

    online_updates = 0
    if online_refine:
        online_updates = _online_refine_memory(
            memory,
            features,
            records,
            support_set,
            tau_update=tau_update,
            tau_new=tau_new,
            alpha=alpha,
            online_max=online_max,
            online_min_cosine=online_min_cosine,
            random_seed=random_seed,
            progress_tracker=progress_tracker,
            log_fn=log_fn,
        )

    open_set_meta = _calibrate_open_set_gates(
        memory,
        features_dir,
        cfg,
        split_task,
        support_set,
        features=features,
        records=records,
        log_fn=log_fn,
    )

    if progress_tracker is not None and progress_tracker.rows:
        last = progress_tracker.rows[-1]
        progress_tracker.snapshot_post_calib(
            int(last.get("samples_seen", 0)),
            int(last.get("online_updates", online_updates)),
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(out_path)

    meta = {
        "features_dir": str(features_dir),
        "n_support": n_support,
        "num_classes": memory.num_classes,
        "class_names": class_names,
        "support_size": len(support_set),
        "support_feature_indices": sorted(int(x) for x in support_set),
        "online_updates": online_updates,
        "online_max": online_max,
        "online_sampling": "known_class_proportional_shuffle",
        "random_seed": random_seed,
        "split_task": split_task,
        "tau_update": tau_update,
        "tau_new": tau_new,
        "alpha": alpha,
        "online_min_cosine": online_min_cosine,
        "support_min_cosine": float(mem_cfg.get("support_min_cosine", 0.15)),
        "open_set_threshold_mode": memory.open_set_threshold_mode,
        "open_set_margin_mode": memory.open_set_margin_mode,
        "global_top1_threshold": (
            None
            if memory.global_top1_threshold is None
            else float(memory.global_top1_threshold.item())
        ),
        "margin_threshold": (
            None
            if memory.margin_threshold is None
            else float(memory.margin_threshold.item())
        ),
    }
    if open_set_meta:
        meta["open_set_calib"] = open_set_meta
    if resume:
        meta["memory_in"] = str(memory_in_path)
        meta["new_classes_added"] = new_classes
        meta["support_updates"] = support_updates

    if progress_tracker is not None and progress_tracker.rows:
        save_build_progress(
            progress_tracker.rows,
            out_path.parent,
            title=f"Build progress ({out_path.parent.name})",
            log_fn=log_fn,
        )
        meta["build_progress_csv"] = str(out_path.parent / "build_progress.csv")
        meta["progress_eval_every_updates"] = progress_every

    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
    log_fn(f"[fit_memory] saved -> {out_path} | classes={memory.num_classes} support={len(support_set)}")
    if resume:
        log_fn(
            f"[fit_memory] resume: new_classes={new_classes} "
            f"support_updates={support_updates} online_updates={online_updates}"
        )
    return support_set
