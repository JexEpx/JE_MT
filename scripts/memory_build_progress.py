"""Learning-curve snapshots during memory build / online refine"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import torch

from src.evaluation import OWODMetrics, get_split, resolve_split_task
from src.memory import UNKNOWN, PrototypeMemory

PROGRESS_FIELDS = [
    "step",
    "phase",
    "classifier",
    "samples_seen",
    "online_updates",
    "memory_vectors",
    "known_accuracy_pct",
    "unknown_recall_pct",
    "harmonic_mean_pct",
    "accuracy_pct",
    "total_eval",
]


def eval_memory_snapshot(
    memory: PrototypeMemory,
    features,
    records: list[dict],
    cfg: dict,
    *,
    support_set: set[int] | None = None,
) -> dict[str, float]:
    split_task = resolve_split_task(cfg)
    known_names = get_split(split_task).known_names if split_task is not None else list(memory.labels)
    known_set = set(known_names)
    metrics = OWODMetrics(known_names)

    correct = total = 0
    for r in records:
        idx = int(r["feature_index"])
        if support_set and idx in support_set:
            continue
        true = r.get("gt_category")
        if not true:
            continue

        z = torch.from_numpy(features[idx]).float().to(memory.device)
        z = memory.prepare_for_inference(z)
        pred, _ = memory.predict(z, prepared=True)

        true_bucket = true if true in known_set else UNKNOWN
        total += 1
        if pred == true_bucket:
            correct += 1
        metrics.add(pred, true)

    owod = metrics.compute()
    acc_pct = 100.0 * correct / max(1, total)
    return {
        "known_accuracy_pct": owod.known_accuracy * 100.0,
        "unknown_recall_pct": owod.unknown_recall * 100.0,
        "harmonic_mean_pct": owod.harmonic_mean * 100.0,
        "accuracy_pct": acc_pct,
        "total_eval": float(total),
    }


def _apply_eval_overrides(memory: PrototypeMemory, cfg: dict) -> None:
    for key in ("use_global_threshold", "use_proto_margin"):
        if key in cfg:
            setattr(memory, key, bool(cfg[key]))
    if "use_mhn_classify" in cfg:
        memory.use_mhn_classify = bool(cfg["use_mhn_classify"])
    if "use_mhn_refine" in cfg:
        memory.use_mhn_refine = bool(cfg["use_mhn_refine"])
    if "mhn_beta" in cfg:
        memory.mhn.beta = float(cfg["mhn_beta"])


class BuildProgressTracker:
    # Record test-set eval snapshots; trigger every N successful online updates

    def __init__(
        self,
        memory: PrototypeMemory,
        features,
        test_records: list[dict],
        eval_cfg: dict,
        support_set: set[int],
        *,
        eval_every_updates: int,
        log_fn: Callable = print,
        calibrate_gates: Optional[Callable[[], None]] = None,
    ) -> None:
        self.memory = memory
        self.features = features
        self.test_records = test_records
        self.eval_cfg = eval_cfg
        self.support_set = support_set
        self.eval_every_updates = max(1, int(eval_every_updates))
        self.log_fn = log_fn
        self.calibrate_gates = calibrate_gates
        self.rows: list[dict] = []
        self._step = 0
        self._saved_eval = {
            "use_mhn_classify": memory.use_mhn_classify,
            "use_mhn_refine": memory.use_mhn_refine,
            "mhn_beta": memory.mhn.beta,
        }

    def record(
        self,
        phase: str,
        samples_seen: int,
        online_updates: int,
        *,
        gates_ready: bool = False,
    ) -> None:
        if self.calibrate_gates is not None and not gates_ready:
            self.calibrate_gates()
        _apply_eval_overrides(self.memory, self.eval_cfg)
        metrics = eval_memory_snapshot(
            self.memory,
            self.features,
            self.test_records,
            self.eval_cfg,
            support_set=self.support_set,
        )
        self.rows.append(
            {
                "step": self._step,
                "phase": phase,
                "classifier": "cosine",
                "samples_seen": samples_seen,
                "online_updates": online_updates,
                "memory_vectors": int(self.memory.prototypes.shape[0]),
                **metrics,
            }
        )
        self.log_fn(
            f"[build_progress] step={self._step:4d} phase={phase:14s} "
            f"samples={samples_seen:5d} updates={online_updates:5d} "
            f"known={metrics['known_accuracy_pct']:5.1f}% "
            f"unk={metrics['unknown_recall_pct']:5.1f}% "
            f"H={metrics['harmonic_mean_pct']:5.1f}%"
        )
        _apply_eval_overrides(self.memory, self._saved_eval)
        self._step += 1

    def snapshot_post_build(self) -> None:
        self.record("post_build", 0, 0)

    def on_online_step(self, samples_seen: int, online_updates: int) -> None:
        if online_updates > 0 and online_updates % self.eval_every_updates == 0:
            self.record("online_refine", samples_seen, online_updates)

    def snapshot_finish(self, samples_seen: int, online_updates: int) -> None:
        if not self.rows:
            return
        last = self.rows[-1]
        if int(last["online_updates"]) == online_updates and last["phase"] == "online_refine":
            return
        self.record("online_refine", samples_seen, online_updates)

    def snapshot_post_calib(self, samples_seen: int, online_updates: int) -> None:
        # Final snapshot after fit-time open-set calib (gates already on memory)
        if (
            self.rows
            and int(self.rows[-1]["online_updates"]) == int(online_updates)
            and self.rows[-1]["phase"] == "online_refine"
        ):
            _apply_eval_overrides(self.memory, self.eval_cfg)
            metrics = eval_memory_snapshot(
                self.memory,
                self.features,
                self.test_records,
                self.eval_cfg,
                support_set=self.support_set,
            )
            row = self.rows[-1]
            row["phase"] = "post_calib"
            row.update(metrics)
            self.log_fn(
                f"[build_progress] step={int(row['step']):4d} phase=post_calib     "
                f"samples={samples_seen:5d} updates={online_updates:5d} "
                f"known={metrics['known_accuracy_pct']:5.1f}% "
                f"unk={metrics['unknown_recall_pct']:5.1f}% "
                f"H={metrics['harmonic_mean_pct']:5.1f}%"
            )
            _apply_eval_overrides(self.memory, self._saved_eval)
            return
        self.record("post_calib", samples_seen, online_updates, gates_ready=True)


def write_progress_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_build_progress(
    rows: list[dict],
    out_pdf: Path,
    *,
    title: str = "Memory build progress",
    x_key: str = "online_updates",
    x_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    x = [int(r[x_key]) for r in rows]
    known = [float(r["known_accuracy_pct"]) for r in rows]
    harmonic = [float(r["harmonic_mean_pct"]) for r in rows]
    unknown = [float(r["unknown_recall_pct"]) for r in rows]

    if x_label is None:
        x_label = "Online memory updates" if x_key == "online_updates" else "Online samples processed"

    fig, ax_left = plt.subplots(figsize=(8, 4.5))
    ax_right = ax_left.twinx()

    line_known, = ax_left.plot(
        x, known, color="#1f77b4", marker="o", markersize=5, linewidth=1.5, label="Known accuracy"
    )
    line_h, = ax_left.plot(
        x, harmonic, color="#d62728", marker="^", markersize=5, linewidth=1.5, label="Harmonic mean"
    )
    post_calib = [i for i, r in enumerate(rows) if r.get("phase") == "post_calib"]
    if post_calib:
        i = post_calib[-1]
        ax_left.scatter(
            [x[i]], [harmonic[i]], color="#d62728", s=120, zorder=5, marker="*",
            label="Final (saved eval)",
        )
    line_unk, = ax_right.plot(
        x, unknown, color="#ff7f0e", marker="s", markersize=4, linewidth=1.2, alpha=0.85, label="Unknown recall"
    )

    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel("Known accuracy / Harmonic mean (%)")
    ax_right.set_ylabel("Unknown recall (%)")
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim(0, 100)
    ax_right.set_ylim(0, 100)

    lines = [line_known, line_h, line_unk]
    ax_left.legend(lines, [ln.get_label() for ln in lines], loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_build_progress(
    rows: list[dict],
    out_dir: Path,
    *,
    title: str = "",
    log_fn: Callable = print,
) -> tuple[Path, Path]:
    csv_path = out_dir / "build_progress.csv"
    pdf_path = out_dir / "build_progress.pdf"
    write_progress_csv(rows, csv_path)
    plot_build_progress(rows, pdf_path, title=title or "Memory build progress")
    log_fn(f"[build_progress] saved CSV -> {csv_path}")
    log_fn(f"[build_progress] saved plot -> {pdf_path}")
    return csv_path, pdf_path
