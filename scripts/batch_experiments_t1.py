"""Batch T1 experiments"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.experiment_runs_t1 import (  
    MEANINGFUL_STAGES,
    build_eval_cfg,
    iter_runs,
    memory_source_for_run,
    meaningful_run_ids,
    resolve_paths,
)
from scripts.evaluate import stage_evaluate  
from scripts.memory import stage_fit_memory  
from scripts.mlp_classifier import (  
    stage_eval_mlp_classifier,
    stage_train_mlp_classifier,
)

DEFAULT_CSV = REPO / "outputs/notebook/experiments/t1_results.csv"

CSV_FIELDS = [
    "run_id",
    "section",
    "feature_pipeline",
    "run_dir",
    "run_kind",
    "status",
    "exemplar_mode",
    "n_support",
    "prototype_init",
    "num_prototypes",
    "exemplar_max_per_class",
    "online_refine",
    "online_max",
    "tau_update",
    "tau_new",
    "alpha",
    "online_min_cosine",
    "open_set_margin_mode",
    "random_seed",
    "use_mhn_classify",
    "use_mhn_refine",
    "mhn_beta",
    "memory_source",
    "update_mode",
    "memory_vectors",
    "online_updates",
    "known_accuracy",
    "unknown_recall",
    "harmonic_mean",
    "a_ose",
    "wilderness_impact",
    "eval_path",
]

_FIT_KEYS = (
    "exemplar_mode", "n_support", "prototype_init", "num_prototypes",
    "exemplar_max_per_class", "online_refine", "online_max",
    "tau_update", "tau_new", "alpha", "online_min_cosine",
    "open_set_margin_mode", "random_seed",
    "use_mhn_classify", "use_mhn_refine", "epochs", "memory_path",
)
_EVAL_KEYS = ("use_mhn_classify", "use_mhn_refine", "mhn_beta")
_METRIC_KEYS = (
    "known_accuracy", "unknown_recall", "harmonic_mean", "a_ose", "wilderness_impact",
)


def _fit_summary(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in _FIT_KEYS}


def _eval_summary(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in _EVAL_KEYS if k in cfg}


def _load_eval_metrics(eval_json: Path) -> dict:
    if not eval_json.is_file():
        return {}
    data = json.loads(eval_json.read_text())
    return {k: data.get(k, "") for k in _METRIC_KEYS}


def _memory_stats(mem_path: Path) -> dict:
    out = {"update_mode": "", "memory_vectors": "", "online_updates": ""}
    if not mem_path.is_file():
        return out
    try:
        import torch

        state = torch.load(mem_path, map_location="cpu")
        out["memory_vectors"] = int(state["prototypes"].shape[0])
        out["update_mode"] = state.get("update_mode", "")
    except Exception:
        pass
    meta_path = mem_path.with_suffix(mem_path.suffix + ".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        out["online_updates"] = meta.get("online_updates", "")
    return out


def _row_base(run: dict, eval_json: Path) -> dict:
    return {
        "run_id": run["id"],
        "section": run["section"],
        "feature_pipeline": run.get("feature_pipeline", "norm"),
        "run_dir": run["run_dir"],
        "run_kind": run.get("run_kind", "memory"),
        "status": "ok",
        "eval_path": str(eval_json),
        "memory_source": "",
        "update_mode": "",
        "memory_vectors": "",
        "online_updates": "",
        **_fit_summary({}),
        **_eval_summary({}),
    }


def _run_mlp(
    repo: Path,
    run: dict,
    *,
    skip_existing: bool,
    eval_only: bool,
    fit_only: bool,
    collect_only: bool,
) -> dict:
    run_dir, _, results_dir = resolve_paths(repo, run)
    eval_json = results_dir / "owod_eval.json"
    ckpt = run_dir / "mlp.pt"
    cfg = run["fit_cfg"](repo)
    row = _row_base(run, eval_json)
    row.update(_fit_summary(cfg))

    if collect_only:
        row.update(_load_eval_metrics(eval_json))
        if not eval_json.is_file():
            row["status"] = "missing_eval"
        return row

    if not eval_only:
        if skip_existing and ckpt.is_file():
            row["status"] = "skipped_train"
        else:
            try:
                print(f"\n[{run['id']}] MLP train -> {ckpt}")
                stage_train_mlp_classifier(run_dir, cfg)
            except Exception as exc:
                row["status"] = f"train_error: {exc}"
                return row

    if not ckpt.is_file():
        row["status"] = "missing_mlp"
        return row

    if fit_only:
        return row

    if skip_existing and eval_json.is_file():
        row["status"] = "skipped_eval"
        row.update(_load_eval_metrics(eval_json))
        return row

    try:
        print(f"[{run['id']}] MLP eval -> {eval_json}")
        stage_eval_mlp_classifier(run_dir, cfg, results_dir)
    except Exception as exc:
        row["status"] = f"eval_error: {exc}"
        return row

    row.update(_load_eval_metrics(eval_json))
    if not eval_json.is_file():
        row["status"] = "missing_eval"
    return row


def _run_memory(
    repo: Path,
    run: dict,
    *,
    skip_existing: bool,
    eval_only: bool,
    fit_only: bool,
    collect_only: bool,
    force_eval: bool,
) -> dict:
    _, mem_path, results_dir = resolve_paths(repo, run)
    eval_json = results_dir / "owod_eval.json"
    row = _row_base(run, eval_json)
    fit_cfg = run["fit_cfg"](repo) if run["fit"] else {}
    row.update(_fit_summary(fit_cfg))
    row.update(_memory_stats(mem_path))
    row["memory_source"] = memory_source_for_run(run, repo)

    eval_cfg = build_eval_cfg(repo, run, mem_path if run["fit"] else None)
    if run["fit"]:
        eval_cfg["memory_path"] = mem_path
    row.update(_eval_summary(eval_cfg))

    if collect_only:
        row.update(_load_eval_metrics(eval_json))
        if not eval_json.is_file():
            row["status"] = "missing_eval"
        return row

    if not eval_only and run["fit"]:
        if skip_existing and mem_path.is_file():
            row["status"] = "skipped_fit"
        else:
            try:
                print(f"\n[{run['id']}] fit -> {mem_path}")
                stage_fit_memory(mem_path, fit_cfg)
            except Exception as exc:
                row["status"] = f"fit_error: {exc}"
                return row
        row.update(_memory_stats(mem_path))

    if fit_only:
        return row

    if run["fit"] and not mem_path.is_file():
        row["status"] = "missing_memory"
        return row

    if skip_existing and eval_json.is_file() and not force_eval:
        row["status"] = "skipped_eval"
        row.update(_load_eval_metrics(eval_json))
        return row

    try:
        print(f"[{run['id']}] eval -> {eval_json}")
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
    run: dict,
    *,
    skip_existing: bool,
    eval_only: bool,
    fit_only: bool,
    collect_only: bool,
    force_eval: bool,
) -> dict:
    flags = {
        "skip_existing": skip_existing,
        "eval_only": eval_only,
        "fit_only": fit_only,
        "collect_only": collect_only,
    }
    if run.get("run_kind") == "mlp_classifier":
        return _run_mlp(repo, run, **flags)
    return _run_memory(repo, run, force_eval=force_eval, **flags)


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
    p = argparse.ArgumentParser(description="Batch T1 memory experiments + CSV summary")
    p.add_argument("--all", action="store_true", help="Run every T1 experiment")
    p.add_argument("--section", type=int, default=None, help="Run all experiments in one section")
    p.add_argument(
        "--run-id",
        action="append",
        nargs="+",
        dest="run_ids",
        metavar="ID",
    )
    p.add_argument(
        "--stage",
        action="append",
        dest="stages",
        metavar="STAGE",
        help="Meaningful stage(s): " + ", ".join(MEANINGFUL_STAGES),
    )
    p.add_argument(
        "--track",
        choices=("proto", "exem", "all"),
        default=None,
        help="With --stage omitted, run all stages in proto/exem/both tracks",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip fit/train; eval only (checkpoint/memory must exist)",
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
    p.add_argument("--output", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--no-merge-csv",
        action="store_true",
        help="Replace output CSV instead of merging rows by run_id",
    )
    args = p.parse_args()

    if (
        not args.all
        and args.section is None
        and not args.run_ids
        and not args.stages
        and args.track is None
    ):
        p.error("Provide --all, --section N, --stage, --track, and/or --run-id ID")

    run_ids = None
    if args.stages:
        run_ids = meaningful_run_ids(*args.stages)
    elif args.track:
        run_ids = meaningful_run_ids(track=args.track)
    if args.run_ids:
        explicit = [rid for group in args.run_ids for rid in group]
        run_ids = explicit if run_ids is None else run_ids + explicit

    section = None if args.all else args.section
    if args.all:
        print("[batch] running all T1 experiments")

    rows: list[dict] = []
    for run in iter_runs(section=section, run_ids=run_ids):
        fp = run.get("feature_pipeline", "norm")
        print(f"=== {run['id']} [{fp}] ({run['run_dir']}) ===")
        rows.append(
            run_one(
                REPO,
                run,
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
            f"({best['run_id']}  {best['run_dir']})"
        )


if __name__ == "__main__":
    main()
