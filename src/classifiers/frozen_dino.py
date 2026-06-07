"""MLP head on frozen DINO crop embeddings with OWOD-style unknown gating"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import UNKNOWN, get_split
from src.memory.open_set_calibration import (
    OpenSetScoreBatch,
    _stratified_subsample_records,
    fit_open_set_thresholds,
)


def build_train_tensors(features, records, split_task: int):
    known = set(get_split(split_task).known_names)
    label_names = sorted(known)
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    xs, ys = [], []
    for r in records:
        name = r.get("gt_category")
        if name not in known:
            continue
        xs.append(int(r["feature_index"]))
        ys.append(label_to_idx[name])
    if not xs:
        raise RuntimeError("no labeled train records for classifier training")
    x = torch.from_numpy(features[xs]).float()
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, label_names


class _MLPHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)


@torch.no_grad()
def _top1_margin_from_probs(probs: torch.Tensor) -> tuple[float, float]:
    if probs.numel() == 1:
        return float(probs.item()), 0.0
    top2 = torch.topk(probs, k=2)
    top1 = float(top2.values[0].item())
    margin = float((top2.values[0] - top2.values[1]).item())
    return top1, margin


@torch.no_grad()
def _classifier_probs(head: nn.Module, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return F.softmax(head(z), dim=-1)[0]


@torch.no_grad()
def collect_classifier_open_set_scores(
    clf: DinoClassifier,
    features,
    records: list[dict],
    known_names: list[str],
    *,
    support_set: set[int] | None = None,
    max_records: int | None = None,
    seed: int = 0,
) -> OpenSetScoreBatch:
    known_set = set(known_names)
    eligible = [
        r
        for r in records
        if r.get("gt_category")
        and (not support_set or int(r["feature_index"]) not in support_set)
    ]
    if max_records and len(eligible) > max_records:
        eligible = _stratified_subsample_records(eligible, max_records, seed=seed)

    if not eligible:
        empty = np.zeros(0, dtype=np.float32)
        return OpenSetScoreBatch(empty, empty, np.zeros(0, dtype=bool))

    clf.head.eval()
    top1_list: list[float] = []
    margin_list: list[float] = []
    unk_flags: list[bool] = []
    labels: list[str] = []
    for r in eligible:
        gt = r["gt_category"]
        z = torch.from_numpy(features[int(r["feature_index"])]).float().to(clf.device)
        probs = _classifier_probs(clf.head, z)
        top1, margin = _top1_margin_from_probs(probs)
        top1_list.append(top1)
        margin_list.append(margin)
        unk_flags.append(gt not in known_set)
        labels.append(gt)

    return OpenSetScoreBatch(
        top1=np.asarray(top1_list, dtype=np.float32),
        margin=np.asarray(margin_list, dtype=np.float32),
        is_unknown_gt=np.asarray(unk_flags, dtype=bool),
        gt_labels=labels,
    )


def fit_classifier_open_set_gates(
    clf: DinoClassifier,
    features,
    records: list[dict],
    *,
    split_task: int,
    cfg: dict,
    log_fn=print,
) -> dict:
    #Fit tau_top1 / tau_margin on calib crops
    known_names = get_split(split_task).known_names
    max_records = cfg.get("open_set_calib_max_records")
    if max_records in (None, 0):
        max_records = None
    else:
        max_records = int(max_records)

    batch = collect_classifier_open_set_scores(
        clf,
        features,
        records,
        known_names,
        max_records=max_records,
        seed=int(cfg.get("open_set_calib_seed", 0)),
    )
    method = str(cfg.get("open_set_calib_method", "known_only"))
    th = fit_open_set_thresholds(
        batch,
        known_names,
        method=method,
        known_percentile=float(cfg.get("open_set_calib_known_percentile", 5.0)),
        known_margin_percentile=float(cfg.get("open_set_calib_known_margin_percentile", 5.0)),
    )
    clf.global_top1_threshold = float(th.global_top1)
    clf.margin_threshold = float(th.margin)
    clf.open_set_calib_method = th.method
    log_fn(
        f"[classifier open_set_calib] method={th.method} n={batch.top1.size} "
        f"tau_top1={th.global_top1:.4f} tau_margin={th.margin:.4f} "
        f"calib_H={th.calib_metrics.get('harmonic_mean', 0):.3f}"
    )
    return {
        "method": th.method,
        "global_top1_threshold": th.global_top1,
        "margin_threshold": th.margin,
        **th.calib_metrics,
    }


def train_classifier(
    features,
    records: list[dict],
    *,
    split_task: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
    log_fn=print,
) -> DinoClassifier:
    x, y, label_names = build_train_tensors(features, records, split_task)
    dim = int(x.shape[1])
    num_classes = len(label_names)

    head = _MLPHead(dim, num_classes, hidden_dim).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = F.normalize(xb.to(device), dim=1)
            yb = yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(head(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        log_fn(
            f"[mlp] epoch {epoch}/{epochs}  loss={total_loss / max(1, n_batches):.4f}"
        )

    return DinoClassifier(
        head=head,
        labels=label_names,
        feature_dim=dim,
        hidden_dim=hidden_dim,
        device=device,
    )


class DinoClassifier:
    # Frozen-DINO embedding MLP with known_only top-1 + margin unknown rejection

    def __init__(
        self,
        *,
        head: nn.Module,
        labels: list[str],
        feature_dim: int,
        hidden_dim: int = 256,
        global_top1_threshold: float | None = None,
        margin_threshold: float | None = None,
        open_set_calib_method: str = "known_only",
        device: torch.device | str = "cpu",
    ):
        self.head = head
        self.labels = list(labels)
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.global_top1_threshold = global_top1_threshold
        self.margin_threshold = margin_threshold
        self.open_set_calib_method = str(open_set_calib_method)
        self.head.to(self.device)
        self.head.eval()

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def _accept_as_known(self, top1: float, margin: float) -> bool:
        tau_g = self.global_top1_threshold
        if tau_g is not None and top1 <= tau_g:
            return False
        tau_m = self.margin_threshold
        if self.num_classes > 1 and tau_m is not None and margin < tau_m:
            return False
        return True

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> tuple[str, float]:
        probs = _classifier_probs(self.head, z.to(self.device))
        i = int(probs.argmax().item())
        c = float(probs[i].item())
        top1, margin = _top1_margin_from_probs(probs)
        if self._accept_as_known(top1, margin):
            return self.labels[i], c
        return UNKNOWN, c

    def to_state_dict(self) -> dict:
        return {
            "head_state": self.head.state_dict(),
            "labels": self.labels,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "global_top1_threshold": self.global_top1_threshold,
            "margin_threshold": self.margin_threshold,
            "open_set_calib_method": self.open_set_calib_method,
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device | str = "cpu") -> DinoClassifier:
        device = torch.device(device)
        labels = list(state["labels"])
        dim = int(state["feature_dim"])
        hidden_dim = int(state["hidden_dim"])
        head = _MLPHead(dim, len(labels), hidden_dim)
        head.load_state_dict(state["head_state"])

        tau_top1 = state["global_top1_threshold"]
        if tau_top1 is not None:
            tau_top1 = float(tau_top1)
        tau_margin = state["margin_threshold"]
        if tau_margin is not None:
            tau_margin = float(tau_margin)

        return cls(
            head=head,
            labels=labels,
            feature_dim=dim,
            hidden_dim=hidden_dim,
            global_top1_threshold=tau_top1,
            margin_threshold=tau_margin,
            open_set_calib_method=str(state["open_set_calib_method"]),
            device=device,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_state_dict(), path)

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> DinoClassifier:
        state = torch.load(path, map_location=device)
        return cls.from_state_dict(state, device=device)


def eval_classifier_on_records(
    clf: DinoClassifier,
    features,
    records: list[dict],
    *,
    split_task: int,
    support_set: set[int] | None = None,
):
    from src.evaluation import OWODMetrics

    known_names = get_split(split_task).known_names
    metrics = OWODMetrics(known_names)

    for r in records:
        idx = int(r["feature_index"])
        if support_set and idx in support_set:
            continue
        true = r.get("gt_category")
        if not true:
            continue
        z = torch.from_numpy(features[idx]).float()
        pred, _ = clf.predict(z)
        metrics.add(pred, true)

    return metrics.compute()
