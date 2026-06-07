"""Frozen-DINO MLP classifier train/eval staging"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.common import resolve_path
from scripts.utils import filter_records_by_split, load_features
from src.classifiers.frozen_dino import (
    DinoClassifier,
    eval_classifier_on_records,
    fit_classifier_open_set_gates,
    train_classifier,
)
from src.evaluation import OWODMetrics, get_split
from src.memory.open_set_defaults import apply_open_set_defaults


def _write_owod_eval(results_dir: Path, owod) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "owod_eval.json"
    path.write_text(json.dumps(owod.to_dict(), indent=2))
    return path


def stage_train_mlp_classifier(run_dir: Path, cfg: dict, log_fn=print) -> Path:
    cfg = apply_open_set_defaults(dict(cfg))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    features_dir = resolve_path(cfg["features_dir"])
    features, records = load_features(features_dir)
    records = filter_records_by_split(records, cfg["train_split_file"], log_fn=log_fn)

    clf = train_classifier(
        features,
        records,
        split_task=int(cfg["split_task"]),
        epochs=int(cfg.get("epochs", 15)),
        batch_size=int(cfg.get("batch_size", 512)),
        lr=float(cfg.get("lr", 1e-3)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        device=device,
        log_fn=log_fn,
    )

    calib_split = cfg.get("open_set_calib_split_file") or cfg["train_split_file"]
    _, cal_records = load_features(features_dir)
    cal_records = filter_records_by_split(cal_records, calib_split, log_fn=log_fn)
    calib = fit_classifier_open_set_gates(
        clf,
        features,
        cal_records,
        split_task=int(cfg["split_task"]),
        cfg=cfg,
        log_fn=log_fn,
    )

    ckpt = run_dir / "mlp.pt"
    clf.save(ckpt)
    meta = {
        "run_kind": "mlp_classifier",
        "checkpoint": str(ckpt),
        "train_split_file": cfg["train_split_file"],
        "open_set_calib_split_file": calib_split,
        "split_task": cfg["split_task"],
        "epochs": cfg.get("epochs", 15),
        "open_set_calib_method": clf.open_set_calib_method,
        "global_top1_threshold": clf.global_top1_threshold,
        "margin_threshold": clf.margin_threshold,
        **calib,
    }
    (run_dir / "mlp.meta.json").write_text(json.dumps(meta, indent=2))
    log_fn(f"[MLP] saved classifier -> {ckpt}")
    return ckpt


def stage_eval_mlp_classifier(run_dir: Path, cfg: dict, results_dir: Path, log_fn=print) -> dict:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ckpt = run_dir / "mlp.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing classifier checkpoint: {ckpt}")

    clf = DinoClassifier.load(ckpt, device=device)
    log_fn(
        f"[MLP] gates method={clf.open_set_calib_method} "
        f"tau_top1={clf.global_top1_threshold} tau_margin={clf.margin_threshold}"
    )

    features, records = load_features(resolve_path(cfg["features_dir"]))
    records = filter_records_by_split(records, cfg["test_split_file"], log_fn=log_fn)
    owod = eval_classifier_on_records(
        clf, features, records, split_task=int(cfg["split_task"])
    )
    known_names = get_split(int(cfg["split_task"])).known_names
    OWODMetrics(known_names).print_report(owod)
    _write_owod_eval(results_dir, owod)
    log_fn(f"[MLP] OWOD metrics -> {results_dir / 'owod_eval.json'}")
    return owod.to_dict()
