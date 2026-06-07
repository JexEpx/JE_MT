""" OWOD Tasks 1–4 cross-task evaluation"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from src.memory.open_set_defaults import apply_open_set_defaults

from scripts.experiment_runs_t1 import (
    E2_09_EXEMPLAR_FIT_KW,
    E2_09_EXEMPLAR_RUN_DIR,
    FEATURE_PIPELINE,
    OPEN_SET_CALIB_MAX_RECORDS,
    P1_18_PROTOTYPE_FIT_KW,
    P1_18_PROTOTYPE_RUN_DIR,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "outputs/notebook/encode_coco_gt_normalized"
IMAGESETS = REPO_ROOT / "data/OWDETR/VOC2007/ImageSets"
TEST_SPLIT = IMAGESETS / "test.txt"
OUT_DIR = "owod_cross_task_incremental"

_P18_DIR_TAIL = P1_18_PROTOTYPE_RUN_DIR.removeprefix("t1_")
_E209_DIR_TAIL = E2_09_EXEMPLAR_RUN_DIR.removeprefix("t1_")

TRACKS = ("mlp", "proto", "exem")
OWOD_ARM_SLOT: dict[str, int] = {"mlp": 1, "proto": 2, "exem": 3}

SHARED_FIT_DEFAULTS = apply_open_set_defaults({
    "test_split_file": str(TEST_SPLIT),
    "device": "cuda",
    "random_seed": 153,
    "feature_pipeline": FEATURE_PIPELINE,
    "progress_eval_every_updates": 0,
    "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
})

PROTO_CHAMPION_BUILD = apply_open_set_defaults({
    **P1_18_PROTOTYPE_FIT_KW,
    **SHARED_FIT_DEFAULTS,
})

EXEM_CHAMPION_BUILD = apply_open_set_defaults({
    **E2_09_EXEMPLAR_FIT_KW,
    **SHARED_FIT_DEFAULTS,
})

MLP_TRAIN_DEFAULTS = apply_open_set_defaults({
    "features_dir": str(FEATURES_DIR),
    "test_split_file": str(TEST_SPLIT),
    "epochs": 15,
    "batch_size": 512,
    "lr": 1e-3,
    "hidden_dim": 256,
    "device": "cuda",
    "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
    "open_set_calib_seed": 0,
})



MHN_BETA_OWOD = 20.0
OWOD_CLOSED_SET_TASKS = {4}
OWOD_ONLINE_MAX_STEP = 5000
OWOD_ONLINE_MAX_CAP = 20_000


def owod_online_max(task: int) -> int:
    # Per-task online refinement cap
    return min(OWOD_ONLINE_MAX_STEP * int(task), OWOD_ONLINE_MAX_CAP)


def owod_mhn_beta(track: str) -> float:
    if track not in ("proto", "exem"):
        raise KeyError(f"owod_mhn_beta only for proto|exem, got {track!r}")
    return MHN_BETA_OWOD


def owod_mhn_kw(track: str) -> dict[str, Any]:
    # MHN classify + MHN gates for both memory fit and MHN eval
    return {
        "use_mhn_classify": True,
        "use_mhn_refine": False,
        "mhn_beta": owod_mhn_beta(track),
        "open_set_gate_use_cosine_class": False,
    }


def closed_set_eval_kw(task: int) -> dict[str, bool]:
    if int(task) not in OWOD_CLOSED_SET_TASKS:
        return {}
    return {
        "use_global_threshold": False,
        "use_proto_margin": False,
    }


def memory_train_split_path(task: int) -> Path:
    # Per-task memory fit / online pool: full OWOD train split for that task
    return IMAGESETS / f"t{task}_train.txt"


def open_set_calib_split_path(repo: Path, task: int) -> Path:
    # Gate calibration split for memory fit / MHN eval refit
    if task == 1:
        return memory_train_split_path(task)
    return ensure_owod_train_cumulative_split(repo, task)


def owod_train_cumulative_split_paths(task: int) -> list[Path]:
    # MLP scratch training
    return [IMAGESETS / f"t{t}_train.txt" for t in range(1, task + 1)]


def ensure_owod_train_cumulative_split(repo: Path, task: int) -> Path:
    # Union split t1_train
    out = out_root(repo) / "splits" / f"t{task}_owod_train_cumulative.txt"
    if out.is_file():
        return out
    ids: set[str] = set()
    lines: list[str] = []
    for p in owod_train_cumulative_split_paths(task):
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s in ids:
                continue
            ids.add(s)
            lines.append(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out


def owod_run_id(task: int, track: str) -> str:
    if track not in OWOD_ARM_SLOT:
        raise KeyError(f"Unknown track {track!r}; choose from {TRACKS}")
    return f"A{task}-{OWOD_ARM_SLOT[track]:02d}"


def owod_run_dir(task: int, track: str) -> str:
    if track == "mlp":
        return f"t{task}_mlp_cumulative"
    if track == "proto":
        return f"t{task}_{_P18_DIR_TAIL}"
    if track == "exem":
        return f"t{task}_{_E209_DIR_TAIL}"
    raise KeyError(track)


def _build_runs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for task in (1, 2, 3, 4):
        for track in TRACKS:
            specs.append({
                "id": owod_run_id(task, track),
                "task": task,
                "track": track,
                "run_dir": owod_run_dir(task, track),
                "resume": track != "mlp" and task > 1,
            })
    return specs


RUNS: list[dict[str, Any]] = _build_runs()


def iter_runs(
    tasks: list[int] | None = None,
    run_ids: list[str] | None = None,
    tracks: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for r in RUNS:
        if tasks is not None and r["task"] not in tasks:
            continue
        if run_ids is not None and r["id"] not in run_ids:
            continue
        if tracks is not None and r["track"] not in tracks:
            continue
        yield r


def out_root(repo: Path) -> Path:
    return (repo / "outputs/notebook/experiments" / OUT_DIR).resolve()


def _prev_run_dir(repo: Path, task: int, track: str) -> Path:
    return out_root(repo) / owod_run_dir(task - 1, track)


def mlp_fit_cfg(repo: Path, task: int) -> dict[str, Any]:
    cum = ensure_owod_train_cumulative_split(repo, task)
    return apply_open_set_defaults({
        **MLP_TRAIN_DEFAULTS,
        "train_split_file": str(cum),
        "open_set_calib_split_file": str(cum),
        "split_task": task,
    })


def memory_fit_cfg(repo: Path, task: int, track: str) -> dict[str, Any]:
    base = EXEM_CHAMPION_BUILD if track == "exem" else PROTO_CHAMPION_BUILD
    train_split = memory_train_split_path(task)
    calib = open_set_calib_split_path(repo, task)
    return apply_open_set_defaults({
        **base,
        **owod_mhn_kw(track),
        "online_max": owod_online_max(task),
        "features_dir": str(FEATURES_DIR),
        "train_split_file": str(train_split),
        "open_set_calib_split_file": str(calib),
        "split_task": task,
    })


def mhn_eval_cfg(
    repo: Path, task: int, memory_path: Path, track: str
) -> dict[str, Any]:
    calib = open_set_calib_split_path(repo, task)
    return apply_open_set_defaults({
        **owod_mhn_kw(track),
        **closed_set_eval_kw(task),
        "features_dir": str(FEATURES_DIR),
        "memory_path": str(memory_path),
        "test_split_file": str(TEST_SPLIT),
        "open_set_calib_split_file": str(calib),
        "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
        "split_task": task,
    })


def mlp_eval_cfg(repo: Path, task: int) -> dict[str, Any]:
    return apply_open_set_defaults({
        "features_dir": str(FEATURES_DIR),
        "test_split_file": str(TEST_SPLIT),
        "split_task": task,
    })


def resolve_paths(
    repo: Path, run: dict[str, Any]
) -> tuple[Path, Path | None, Path, Path | None]:
    run_dir = out_root(repo) / run["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = (run_dir / "eval" / "eval").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    track = run["track"]
    if track == "mlp":
        return run_dir, None, results_dir, None

    mem_path = run_dir / "memory.pt"
    memory_in: Path | None = None
    if run.get("resume") and run["task"] > 1:
        prev_mem = _prev_run_dir(repo, run["task"], track) / "memory.pt"
        memory_in = prev_mem if prev_mem.is_file() else None
    return run_dir, mem_path, results_dir, memory_in
