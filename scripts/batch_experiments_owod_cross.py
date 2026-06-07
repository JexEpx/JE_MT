"""OWOD Tasks 1–4"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.experiment_runs_owod_cross import (
    EXEM_CHAMPION_BUILD,
    PROTO_CHAMPION_BUILD,
    iter_runs,
    memory_fit_cfg,
    mhn_eval_cfg,
    mlp_eval_cfg,
    mlp_fit_cfg,
    resolve_paths,
)
from scripts.evaluate import stage_evaluate 
from scripts.memory import stage_fit_memory  
from scripts.mlp_classifier import (
    stage_eval_mlp_classifier,
    stage_train_mlp_classifier,
)

DEFAULT_CSV = REPO / "outputs/notebook/experiments/A_owod.csv"

CSV_FIELDS = [
    "run_id",
    "task",
    "track",
    "run_dir",
    "status",
    "train_split",
    "split_task",
    "n_support",
    "prototype_init",
    "exemplar_mode",
    "online_max",
    "use_mhn_classify",
    "mhn_beta",
    "memory_vectors",
    "online_updates",
    "new_classes_added",
    "known_accuracy",
    "unknown_recall",
    "harmonic_mean",
    "a_ose",
    "wilderness_impact",
    "eval_path",
]


def _memory_stats(mem_path: Path) -> dict:
    out = {
        "memory_vectors": "",
        "online_updates": "",
        "new_classes_added": "",
    }
    if not mem_path.is_file():
        return out
    try:
        import torch

        state = torch.load(mem_path, map_location="cpu")
        out["memory_vectors"] = int(state["prototypes"].shape[0])
    except Exception:
        pass
    meta_path = mem_path.with_suffix(mem_path.suffix + ".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        out["online_updates"] = meta.get("online_updates", "")
        out["new_classes_added"] = meta.get("new_classes_added", "")
    return out


def _load_eval_metrics(eval_json: Path) -> dict:
    if not eval_json.is_file():
        return {}
    data = json.loads(eval_json.read_text())
    return {
        "known_accuracy": data.get("known_accuracy", ""),
        "unknown_recall": data.get("unknown_recall", ""),
        "harmonic_mean": data.get("harmonic_mean", ""),
        "a_ose": data.get("a_ose", ""),
        "wilderness_impact": data.get("wilderness_impact", ""),
    }


def _row_base(spec: dict, eval_json: Path) -> dict:
    task = spec["task"]
    track = spec["track"]
    build = EXEM_CHAMPION_BUILD if track == "exem" else PROTO_CHAMPION_BUILD
    return {
        "run_id": spec["id"],
        "task": task,
        "track": track,
        "run_dir": spec["run_dir"],
        "status": "ok",
        "split_task": task,
        "eval_path": str(eval_json),
        "train_split": "",
        "n_support": "" if track == "mlp" else build.get("n_support", ""),
        "prototype_init": "" if track == "mlp" else build.get("prototype_init", ""),
        "exemplar_mode": "" if track == "mlp" else build.get("exemplar_mode", False),
        "online_max": "",
        "use_mhn_classify": "",
        "mhn_beta": "",
        "memory_vectors": "",
        "online_updates": "",
        "new_classes_added": "",
    }


def _run_mlp(
    repo: Path,
    spec: dict,
    *,
    skip_existing: bool,
    eval_only: bool,
    fit_only: bool,
    collect_only: bool,
) -> dict:
    run_dir, _, results_dir, _ = resolve_paths(repo, spec)
    eval_json = results_dir / "owod_eval.json"
    mlp_ckpt = run_dir / "mlp.pt"
    row = _row_base(spec, eval_json)

    fit_cfg = mlp_fit_cfg(repo, spec["task"])
    row["train_split"] = Path(fit_cfg["train_split_file"]).name

    if collect_only:
        row.update(_load_eval_metrics(eval_json))
        if not eval_json.is_file():
            row["status"] = "missing_eval"
        return row

    if not eval_only:
        if skip_existing and mlp_ckpt.is_file():
            row["status"] = "skipped_train"
        else:
            try:
                print(f"\n[{spec['id']}] MLP train (scratch) -> {mlp_ckpt}")
                stage_train_mlp_classifier(run_dir, fit_cfg)
            except Exception as exc:
                row["status"] = f"train_error: {exc}"
                return row

    if not mlp_ckpt.is_file():
        row["status"] = "missing_mlp"
        return row

    if fit_only:
        return row

    if skip_existing and eval_json.is_file():
        row["status"] = "skipped_eval"
        row.update(_load_eval_metrics(eval_json))
        return row

    try:
        print(f"[{spec['id']}] MLP eval -> {eval_json}")
        stage_eval_mlp_classifier(run_dir, mlp_eval_cfg(repo, spec["task"]), results_dir)
    except Exception as exc:
        row["status"] = f"eval_error: {exc}"
        return row

    row.update(_load_eval_metrics(eval_json))
    if not eval_json.is_file():
        row["status"] = "missing_eval"
    return row


def _run_memory_mhn(
    repo: Path,
    spec: dict,
    *,
    skip_existing: bool,
    eval_only: bool,
    fit_only: bool,
    collect_only: bool,
    force_eval: bool,
) -> dict:
    run_dir, mem_path, results_dir, memory_in = resolve_paths(repo, spec)
    assert mem_path is not None
    eval_json = results_dir / "owod_eval.json"
    row = _row_base(spec, eval_json)
    fit_cfg = memory_fit_cfg(repo, spec["task"], spec["track"])
    row["train_split"] = Path(fit_cfg["train_split_file"]).name
    row["online_max"] = fit_cfg.get("online_max", "")
    eval_cfg = mhn_eval_cfg(repo, spec["task"], mem_path, spec["track"])
    row["use_mhn_classify"] = eval_cfg.get("use_mhn_classify", True)
    row["mhn_beta"] = eval_cfg.get("mhn_beta", "")

    if collect_only:
        row.update(_memory_stats(mem_path))
        row.update(_load_eval_metrics(eval_json))
        if not eval_json.is_file():
            row["status"] = "missing_eval"
        return row

    if not eval_only:
        if spec.get("resume"):
            if memory_in is None or not memory_in.is_file():
                row["status"] = f"missing_memory_in: {memory_in}"
                return row
        if skip_existing and mem_path.is_file():
            row["status"] = "skipped_fit"
        else:
            try:
                if spec.get("resume"):
                    print(
                        f"\n[{spec['id']}] resume {memory_in.name} -> {mem_path} "
                        f"({row['train_split']})"
                    )
                else:
                    print(
                        f"\n[{spec['id']}] memory fit -> {mem_path} ({row['train_split']})"
                    )
                stage_fit_memory(mem_path, fit_cfg, memory_in=memory_in)
            except Exception as exc:
                row["status"] = f"fit_error: {exc}"
                return row

    row.update(_memory_stats(mem_path))

    if not mem_path.is_file():
        row["status"] = "missing_memory"
        return row

    if fit_only:
        return row

    if skip_existing and eval_json.is_file() and not force_eval:
        row["status"] = "skipped_eval"
        row.update(_load_eval_metrics(eval_json))
        return row

    try:
        print(f"[{spec['id']}] MHN eval -> {eval_json}")
        stage_evaluate(eval_cfg, results_dir, support_set=None)
    except Exception as exc:
        row["status"] = f"eval_error: {exc}"
        return row

    row.update(_load_eval_metrics(eval_json))
    if not eval_json.is_file():
        row["status"] = "missing_eval"
    return row


def run_one(
    repo: Path,
    spec: dict,
    *,
    skip_existing: bool,
    eval_only: bool,
    fit_only: bool,
    collect_only: bool,
    force_eval: bool,
) -> dict:
    if spec["track"] == "mlp":
        return _run_mlp(
            repo,
            spec,
            skip_existing=skip_existing,
            eval_only=eval_only,
            fit_only=fit_only,
            collect_only=collect_only,
        )
    return _run_memory_mhn(
        repo,
        spec,
        skip_existing=skip_existing,
        eval_only=eval_only,
        fit_only=fit_only,
        collect_only=collect_only,
        force_eval=force_eval,
    )


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], out_path: Path, *, merge: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = rows
    if merge and out_path.is_file():
        by_id = {r["run_id"]: r for r in _read_csv_rows(out_path) if r.get("run_id")}
        for row in rows:
            by_id[row["run_id"]] = row
        merged = [by_id[k] for k in sorted(by_id)]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    if merge and len(merged) > len(rows):
        print(f"\nWrote {len(merged)} rows ({len(rows)} updated) -> {out_path}")
    else:
        print(f"\nWrote {len(merged)} rows -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="OWOD track A: MLP vs proto vs exem")
    p.add_argument("--task", type=int, nargs="+", dest="tasks")
    p.add_argument(
        "--track",
        "--tracks",
        choices=["mlp", "proto", "exem"],
        action="append",
        dest="tracks",
    )
    p.add_argument(
        "--run-id",
        action="append",
        nargs="+",
        dest="run_ids",
        metavar="ID",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip fit/train; MHN/MLP eval only (memory/MLP checkpoint must exist)",
    )
    p.add_argument(
        "--fit-only",
        action="store_true",
        help="Memory fit or MLP train only; skip eval",
    )
    p.add_argument("--collect-only", action="store_true")
    p.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run eval even if owod_eval.json exists (use with --skip-existing)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV path (default: {DEFAULT_CSV.relative_to(REPO)})",
    )
    p.add_argument(
        "--no-merge-csv",
        action="store_true",
        help="Replace output CSV instead of merging rows by run_id",
    )
    args = p.parse_args()

    run_ids = None
    if args.run_ids:
        run_ids = [rid for group in args.run_ids for rid in group]

    rows: list[dict] = []
    for spec in iter_runs(tasks=args.tasks, run_ids=run_ids, tracks=args.tracks):
        print(f"=== {spec['id']}  {spec['track']}  {spec['run_dir']} ===")
        rows.append(
            run_one(
                REPO,
                spec,
                skip_existing=args.skip_existing,
                eval_only=args.eval_only,
                fit_only=args.fit_only,
                collect_only=args.collect_only,
                force_eval=args.force_eval,
            )
        )

    write_csv(rows, args.output.resolve(), merge=not args.no_merge_csv)

    scored = [
        r for r in rows
        if isinstance(r.get("harmonic_mean"), (int, float))
    ]
    if scored:
        best = max(scored, key=lambda r: r["harmonic_mean"])
        print(
            f"Best H: {best['harmonic_mean']:.4f}  "
            f"({best['run_id']}  task={best['task']}  {best['track']})"
        )


if __name__ == "__main__":
    main()
