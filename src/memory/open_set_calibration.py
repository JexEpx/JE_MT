"""Known-only open-set threshold calibration (top-1 + margin + optional Mahalanobis)"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.evaluation import OWODMetrics, UNKNOWN

if TYPE_CHECKING:
    from src.memory.prototype_memory import PrototypeMemory


@dataclass
class OpenSetScoreBatch:
    # Per-crop top-1 cosine, margin, optional Mahalanobis distance

    top1: np.ndarray
    margin: np.ndarray
    is_unknown_gt: np.ndarray
    gt_labels: list[str] = field(default_factory=list)
    maha: np.ndarray | None = None

    @property
    def known_top1(self) -> np.ndarray:
        return self.top1[~self.is_unknown_gt]

    @property
    def known_margin(self) -> np.ndarray:
        return self.margin[~self.is_unknown_gt]

    @property
    def known_maha(self) -> np.ndarray:
        if self.maha is None:
            return np.zeros(0, dtype=np.float32)
        return self.maha[~self.is_unknown_gt]


@dataclass
class OpenSetThresholds:
    global_top1: float
    margin: float
    method: str
    percentile: float = 95.0
    calib_metrics: dict = field(default_factory=dict)
    maha: float | None = None


def _stratified_subsample_records(
    records: list[dict],
    max_records: int,
    *,
    seed: int = 0,
) -> list[dict]:
    # Sample by gt_category, unknown-side estimates carry more sampling noise under a cap
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r["gt_category"])].append(r)

    categories = sorted(groups)
    n_groups = len(categories)
    base = max_records // n_groups
    remainder = max_records % n_groups

    rng = np.random.default_rng(seed)
    sampled: list[dict] = []
    for i, cat in enumerate(categories):
        cap = base + (1 if i < remainder else 0)
        pool = groups[cat]
        if len(pool) <= cap:
            sampled.extend(pool)
        else:
            pick = rng.choice(len(pool), size=cap, replace=False)
            sampled.extend(pool[int(j)] for j in pick)
    return sampled


def collect_open_set_scores(
    memory: PrototypeMemory,
    features,
    records: list[dict],
    known_names: list[str],
    *,
    support_set: set[int] | None = None,
    max_records: int | None = None,
    seed: int = 0,
) -> OpenSetScoreBatch:
    known_set = set(known_names)
    eligible: list[dict] = []
    for r in records:
        idx = int(r["feature_index"])
        if support_set and idx in support_set:
            continue
        gt = r.get("gt_category")
        if not gt:
            continue
        eligible.append(r)

    if max_records is not None and max_records > 0 and len(eligible) > max_records:
        eligible = _stratified_subsample_records(eligible, max_records, seed=seed)

    top1_list: list[float] = []
    margin_list: list[float] = []
    maha_list: list[float] | None = [] if memory._uses_maha_gate() else None
    unk_flags: list[bool] = []
    labels: list[str] = []

    for r in eligible:
        idx = int(r["feature_index"])
        gt = r["gt_category"]
        z = torch.from_numpy(features[idx]).float().to(memory.device)
        top1, margin, maha = memory.collect_open_set_signal_and_maha(z)
        top1_list.append(top1)
        margin_list.append(margin)
        if maha_list is not None and maha is not None:
            maha_list.append(maha)
        unk_flags.append(gt not in known_set)
        labels.append(gt)

    if not top1_list:
        return OpenSetScoreBatch(
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
            maha=np.zeros(0, dtype=np.float32) if maha_list is not None else None,
        )

    return OpenSetScoreBatch(
        top1=np.asarray(top1_list, dtype=np.float32),
        margin=np.asarray(margin_list, dtype=np.float32),
        is_unknown_gt=np.asarray(unk_flags, dtype=bool),
        gt_labels=labels,
        maha=np.asarray(maha_list, dtype=np.float32) if maha_list is not None else None,
    )


def _predict_known_mask(
    top1: np.ndarray,
    margin: np.ndarray,
    tau_g: float,
    tau_m: float,
    *,
    maha: np.ndarray | None = None,
    tau_maha: float | None = None,
    gate_mode: str = "cosine_margin",
) -> np.ndarray:
    mode = str(gate_mode or "cosine_margin")
    mask = np.ones(top1.shape[0], dtype=bool)
    if mode in ("cosine_margin", "cosine_margin_maha"):
        mask &= (top1 > tau_g) & (margin >= tau_m)
    if mode in ("maha", "cosine_margin_maha"):
        if maha is None or tau_maha is None:
            mask &= False
        else:
            mask &= maha <= tau_maha
    return mask


def _eval_calib_metrics(
    batch: OpenSetScoreBatch,
    known_names: list[str],
    tau_g: float,
    tau_m: float,
    *,
    tau_maha: float | None = None,
    gate_mode: str = "cosine_margin",
) -> dict[str, float]:
    metrics = OWODMetrics(known_names)
    for i, gt in enumerate(batch.gt_labels):
        is_unk = bool(batch.is_unknown_gt[i])
        maha_i = None if batch.maha is None else batch.maha[i : i + 1]
        pred_known = bool(
            _predict_known_mask(
                batch.top1[i : i + 1],
                batch.margin[i : i + 1],
                tau_g,
                tau_m,
                maha=maha_i,
                tau_maha=tau_maha,
                gate_mode=gate_mode,
            )[0]
        )
        # Gate-only eval: accepted crops count as correct class; rejected → UNKNOWN.
        pred = UNKNOWN if not pred_known else gt
        metrics.add(pred, gt)
    r = metrics.compute()
    return {
        "known_accuracy": float(r.known_accuracy),
        "unknown_recall": float(r.unknown_recall),
        "harmonic_mean": float(r.harmonic_mean),
        "open_set_error_rate": float(r.open_set_error_rate),
    }


def fit_known_only_thresholds(
    batch: OpenSetScoreBatch,
    *,
    known_percentile: float = 5.0,
    known_margin_percentile: float = 5.0,
    maha_keep_percentile: float = 95.0,
    min_top1: float = 0.0,
    min_margin: float = 0.0,
    fallback_top1: float = 0.32,
    fallback_margin: float = 0.07,
    fallback_maha: float = 10.0,
    gate_mode: str = "cosine_margin",
) -> OpenSetThresholds:

    # Set thresholds from the KNOWN-class score distribution only
    p = float(known_percentile)
    p_m = float(known_margin_percentile)
    kn_t1 = batch.known_top1
    kn_m = batch.known_margin
    kn_maha = batch.known_maha

    tau_g = float(np.percentile(kn_t1, p)) if kn_t1.size > 0 else float(fallback_top1)
    tau_m = float(np.percentile(kn_m, p_m)) if kn_m.size > 0 else float(fallback_margin)

    tau_maha = None
    if gate_mode in ("maha", "cosine_margin_maha"):
        if kn_maha.size > 0:
            tau_maha = float(np.percentile(kn_maha, float(maha_keep_percentile)))
        else:
            tau_maha = float(fallback_maha)

    tau_g = max(min_top1, tau_g)
    tau_m = max(min_margin, tau_m)

    return OpenSetThresholds(
        global_top1=tau_g,
        margin=tau_m,
        method="known_only",
        percentile=p,
        maha=tau_maha,
    )


def fit_open_set_thresholds(
    batch: OpenSetScoreBatch,
    known_names: list[str],
    *,
    method: str = "known_only",
    maha_percentile: float = 95.0,
    known_percentile: float = 5.0,
    known_margin_percentile: float = 5.0,
    memory: PrototypeMemory | None = None,
) -> OpenSetThresholds:
    if method != "known_only":
        raise ValueError(
            f"unsupported open-set calibration method: {method!r} (use known_only)"
        )
    gate_mode = getattr(memory, "open_set_gate_mode", "cosine_margin") if memory else "cosine_margin"
    th = fit_known_only_thresholds(
        batch,
        known_percentile=known_percentile,
        known_margin_percentile=known_margin_percentile,
        maha_keep_percentile=maha_percentile,
        gate_mode=gate_mode,
    )
    th.calib_metrics = _eval_calib_metrics(
        batch,
        known_names,
        th.global_top1,
        th.margin,
        tau_maha=th.maha,
        gate_mode=gate_mode,
    )
    return th
