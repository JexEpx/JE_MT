"""T1 experiment runs and config builders"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from src.memory.open_set_defaults import (
    OPEN_SET_CALIB_MAX_RECORDS,
    apply_open_set_defaults,
)

CHAMPION = {
    "prototype": "t1_proto_s32_k3_km_on5k_tu085_tn040_omc25",
    "exemplar": "t1_exem_cap50_on5k_tu085_tn040_omc25",
}

FEATURE_PIPELINE = "norm"
FEATURES_DIR = "outputs/notebook/encode_coco_gt_normalized"
EXP_ROOT = "outputs/notebook/experiments"
IMAGESETS = "data/OWDETR/VOC2007/ImageSets"
T1_TRAIN = f"{IMAGESETS}/t1_train.txt"
TEST_SPLIT = f"{IMAGESETS}/test.txt"

MHN_BETA_CHAMPION = 20.0
MHN_BETA_S3_COMPARE = 30.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split(repo: Path, rel: str) -> str:
    return str(repo / rel)


def _features_dir(repo: Path) -> Path:
    return repo / FEATURES_DIR


def _memory_path(
    repo: Path,
    *,
    memory_path: Path | None,
    memory_key: str,
    memory_run_dir: str | None,
) -> Path:
    if memory_path is not None:
        return memory_path
    if memory_run_dir is not None:
        return repo / EXP_ROOT / memory_run_dir / "memory.pt"
    return repo / EXP_ROOT / CHAMPION[memory_key] / "memory.pt"


def _base_fit(repo: Path) -> dict[str, Any]:
    return apply_open_set_defaults({
        "features_dir": _features_dir(repo),
        "feature_pipeline": FEATURE_PIPELINE,
        "train_split_file": _split(repo, T1_TRAIN),
        "test_split_file": _split(repo, TEST_SPLIT),
        "open_set_calib_split_file": _split(repo, T1_TRAIN),
        "split_task": 1,
        "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
        "n_support": 24,
        "online_refine": False,
        "online_max": 0,
        "tau_update": 0.7,
        "tau_new": 0.5,
        "alpha": 0.1,
        "exemplar_mode": False,
        "exemplar_max_per_class": 50,
        "num_prototypes": 3,
        "prototype_init": "kmeans",
        "device": "cuda",
        "random_seed": 153,
        "use_mhn_classify": False,
        "use_mhn_refine": False,
        "mhn_beta": MHN_BETA_CHAMPION,
        "support_min_cosine": 0.15,
        "online_min_cosine": 0.15,
    })


def _base_eval(repo: Path, memory_path: Path) -> dict[str, Any]:
    return apply_open_set_defaults({
        "features_dir": _features_dir(repo),
        "feature_pipeline": FEATURE_PIPELINE,
        "memory_path": memory_path,
        "train_split_file": _split(repo, T1_TRAIN),
        "open_set_calib_split_file": _split(repo, T1_TRAIN),
        "test_split_file": _split(repo, TEST_SPLIT),
        "split_task": 1,
        "device": "cuda",
        "use_mhn_classify": False,
        "use_mhn_refine": False,
        "mhn_beta": MHN_BETA_CHAMPION,
        "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
        "open_set_calib_seed": 0,
    })


def _fit(overrides: dict) -> Callable[[Path], dict[str, Any]]:
    def builder(repo: Path) -> dict[str, Any]:
        cfg = _base_fit(repo)
        cfg.update(overrides)
        return cfg

    return builder


def _make_eval(
    overrides: dict | None = None,
    *,
    memory_key: str = "prototype",
    memory_run_dir: str | None = None,
) -> Callable[[Path, Path | None], dict[str, Any]]:
    eval_kw = dict(overrides or {})
    if eval_kw.get("use_mhn_classify") and "open_set_gate_use_cosine_class" not in eval_kw:
        eval_kw["open_set_gate_use_cosine_class"] = False

    def builder(repo: Path, memory_path: Path | None = None) -> dict[str, Any]:
        mem = _memory_path(
            repo,
            memory_path=memory_path,
            memory_key=memory_key,
            memory_run_dir=memory_run_dir,
        )
        cfg = _base_eval(repo, mem)
        cfg.update({k: v for k, v in eval_kw.items() if k != "memory_path"})
        return cfg

    return builder


def _mlp_classifier_cfg(repo: Path) -> dict[str, Any]:
    return apply_open_set_defaults({
        "features_dir": str(repo / FEATURES_DIR),
        "train_split_file": _split(repo, T1_TRAIN),
        "test_split_file": _split(repo, TEST_SPLIT),
        "open_set_calib_split_file": _split(repo, T1_TRAIN),
        "split_task": 1,
        "epochs": 15,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_dim": 256,
        "device": "cuda",
        "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
        "open_set_calib_seed": 0,
    })


def _run(
    run_id: str,
    section: int,
    run_dir: str,
    *,
    fit: bool,
    fit_kw: dict | None = None,
    eval_kw: dict | None = None,
    memory_key: str = "prototype",
    memory_run_dir: str | None = None,
    run_kind: str = "memory",
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": run_id,
        "section": section,
        "feature_pipeline": FEATURE_PIPELINE,
        "run_dir": run_dir,
        "fit": fit,
        "run_kind": run_kind,
    }
    if fit_kw is not None:
        spec["fit_cfg"] = _fit(fit_kw)
    if fit:
        spec["eval_cfg"] = _make_eval(eval_kw)
    else:
        spec["memory_key"] = memory_key
        if memory_run_dir is not None:
            spec["memory_run_dir"] = memory_run_dir
        spec["eval_cfg"] = _make_eval(
            eval_kw,
            memory_key=memory_key,
            memory_run_dir=memory_run_dir,
        )
    return spec


_PROTO_KW = {"num_prototypes": 3}

_ONLINE_GATE_KW = {
    "tau_update": 0.8,
    "tau_new": 0.45,
    "alpha": 0.05,
    "online_min_cosine": 0.20,
}

_CHAMPION_PROTOTYPE_KW: dict[str, Any] = {
    **_PROTO_KW,
    "n_support": 32,
    "prototype_init": "kmeans",
}

_ON5K_KM = {
    **_CHAMPION_PROTOTYPE_KW,
    "online_refine": True,
    "online_max": 5000,
}
_ON5K_GATE_GRID = {**_ON5K_KM, **_ONLINE_GATE_KW}

_P18_UPDATE_GATE_KW = {
    "tau_update": 0.85,
    "tau_new": 0.40,
    "alpha": 0.05,
    "online_min_cosine": 0.25,
}

_ON10K_KM = {**_ON5K_KM, "online_max": 10_000}
_ON50K_KM = {**_ON5K_KM, "online_max": 50_000}
_ONALL_KM = {**_ON5K_KM, "online_max": 0, "progress_eval_every_updates": 5000}

GATING_ABLATION_RUN_IDS: list[str] = [
    "P1-03", "P1-07", "P1-08",
    "E2-02", "E2-04", "E2-05",
    "P1-09", "P1-12", "P1-13", "P1-14", "P1-15", "P1-16", "P1-17", "P1-18",
]

_MAHA_REPLACE_GATES = {
    "open_set_maha_mode": "diagonal",
    "open_set_maha_distance": "min_class",
    "open_set_calib_maha_percentile": 95.0,
    "open_set_gate_mode": "maha",
    "use_global_threshold": False,
    "use_proto_margin": False,
}

_SECTION1_SPECS: list[tuple[str, str, dict]] = [
    ("01", "proto_s24_k3_km_noOnline", {**_PROTO_KW, "n_support": 24, "prototype_init": "kmeans", "online_refine": False}),
    ("02", "proto_s24_k3_ex_noOnline", {**_PROTO_KW, "n_support": 24, "prototype_init": "examples", "online_refine": False}),
    ("03", "proto_s32_k3_km_noOnline", {**_PROTO_KW, "n_support": 32, "prototype_init": "kmeans", "online_refine": False}),
    ("04", "proto_s32_k3_ex_noOnline", {**_PROTO_KW, "n_support": 32, "prototype_init": "examples", "online_refine": False}),
    ("05", "proto_s48_k3_km_noOnline", {**_PROTO_KW, "n_support": 48, "prototype_init": "kmeans", "online_refine": False}),
    ("06", "proto_s48_k3_ex_noOnline", {**_PROTO_KW, "n_support": 48, "prototype_init": "examples", "online_refine": False}),
    ("07", "proto_s32_k3_km_maha_noOnline", {**_PROTO_KW, "n_support": 32, "prototype_init": "kmeans", "online_refine": False, **_MAHA_REPLACE_GATES}),
    ("08", "proto_s32_k3_km_globalProto_noOnline", {**_PROTO_KW, "n_support": 32, "prototype_init": "kmeans", "online_refine": False, "open_set_margin_mode": "global_proto"}),
    ("09", "proto_s32_k3_km_on5k", {**_ON5K_KM}),
    ("10", "proto_s32_k3_km_on10k", {**_ON10K_KM}),
    ("11", "proto_s32_k3_km_on50k", {**_ON50K_KM}),
    ("all", "proto_s32_k3_km_onAll_ev5k", {**_ONALL_KM}),
    ("12", "proto_s32_k3_km_on5k_tu085", {**_ON5K_GATE_GRID, "tau_update": 0.85}),
    ("13", "proto_s32_k3_km_on5k_tn040", {**_ON5K_GATE_GRID, "tau_new": 0.40}),
    ("14", "proto_s32_k3_km_on5k_tu085_tn040", {**_ON5K_GATE_GRID, "tau_update": 0.85, "tau_new": 0.40}),
    ("15", "proto_s32_k3_km_on5k_a003", {**_ON5K_GATE_GRID, "alpha": 0.03}),
    ("16", "proto_s32_k3_km_on5k_omc25", {**_ON5K_GATE_GRID, "online_min_cosine": 0.25}),
    ("17", "proto_s32_k3_km_on5k_tu085_tn040_a003", {**_ON5K_GATE_GRID, "tau_update": 0.85, "tau_new": 0.40, "alpha": 0.03}),
    ("18", "proto_s32_k3_km_on5k_tu085_tn040_omc25", {**_ON5K_GATE_GRID, **_P18_UPDATE_GATE_KW}),
]

P1_18_PROTOTYPE_RUN_DIR = "t1_proto_s32_k3_km_on5k_tu085_tn040_omc25"
P1_18_PROTOTYPE_FIT_KW: dict[str, Any] = {**_ON5K_GATE_GRID, **_P18_UPDATE_GATE_KW}


def _exemplar_fit_kw(
    cap: int,
    *,
    online_refine: bool,
    online_max: int = 0,
    tau_update: float = 0.7,
    tau_new: float = 0.5,
) -> dict[str, Any]:
    return {
        "exemplar_mode": True,
        "exemplar_max_per_class": cap,
        "online_refine": online_refine,
        "online_max": online_max,
        "tau_update": tau_update,
        "tau_new": tau_new,
        "n_support": 5,
    }


_EXEMPLAR_SPECS: list[tuple[str, str, dict]] = [
    ("E2-01", "exem_cap20_noOnline", _exemplar_fit_kw(20, online_refine=False)),
    ("E2-02", "exem_cap50_noOnline", _exemplar_fit_kw(50, online_refine=False)),
    ("E2-03", "exem_cap100_noOnline", _exemplar_fit_kw(100, online_refine=False)),
    ("E2-04", "exem_cap50_maha_noOnline", {**_exemplar_fit_kw(50, online_refine=False), **_MAHA_REPLACE_GATES}),
    ("E2-05", "exem_cap50_globalProto_noOnline", {**_exemplar_fit_kw(50, online_refine=False), "open_set_margin_mode": "global_proto"}),
    ("E2-06", "exem_cap50_on5k", _exemplar_fit_kw(50, online_refine=True, online_max=5000)),
    ("E2-07", "exem_cap50_on10k", _exemplar_fit_kw(50, online_refine=True, online_max=10_000)),
    ("E2-08", "exem_cap50_on50k", _exemplar_fit_kw(50, online_refine=True, online_max=50_000)),
    ("E2-09", "exem_cap50_on5k_tu085_tn040_omc25", {
        **_exemplar_fit_kw(50, online_refine=True, online_max=5000),
        **_P18_UPDATE_GATE_KW,
    }),
]

E2_09_EXEMPLAR_RUN_DIR = "t1_exem_cap50_on5k_tu085_tn040_omc25"
E2_09_EXEMPLAR_FIT_KW: dict[str, Any] = {
    **_exemplar_fit_kw(50, online_refine=True, online_max=5000),
    **_P18_UPDATE_GATE_KW,
}

_MHN_BETA_SWEEP = [
    ("05", "proto", "prototype", 5.0, "b05"),
    ("06", "proto", "prototype", 10.0, "b10"),
    ("07", "proto", "prototype", 20.0, "b20"),
    ("08", "exem", "exemplar", 5.0, "b05"),
    ("09", "exem", "exemplar", 10.0, "b10"),
    ("10", "exem", "exemplar", 20.0, "b20"),
]

_REFINE_EVAL = {
    "use_mhn_refine": True,
    "mhn_beta": MHN_BETA_CHAMPION,
}


def _build_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    for suffix, dir_tail, fit_kw in _SECTION1_SPECS:
        runs.append(_run(f"P1-{suffix}", 1, f"t1_{dir_tail}", fit=True, fit_kw=fit_kw))

    for run_id, dir_tail, fit_kw in _EXEMPLAR_SPECS:
        runs.append(_run(run_id, 2, f"t1_{dir_tail}", fit=True, fit_kw=fit_kw))

    runs.extend([
        _run("S3-01", 3, "t1_s3_proto_cosine", fit=False, memory_key="prototype"),
        _run("S3-02", 3, "t1_s3_proto_mhn", fit=False, memory_key="prototype",
             eval_kw={"use_mhn_classify": True, "mhn_beta": MHN_BETA_S3_COMPARE}),
        _run("S3-03", 3, "t1_s3_exem_cosine", fit=False, memory_key="exemplar"),
        _run("S3-04", 3, "t1_s3_exem_mhn", fit=False, memory_key="exemplar",
             eval_kw={"use_mhn_classify": True, "mhn_beta": MHN_BETA_S3_COMPARE}),
    ])

    for suffix, track, memory_key, beta, tag in _MHN_BETA_SWEEP:
        runs.append(_run(
            f"S3-{suffix}", 3, f"t1_s3_{track}_mhn_{tag}", fit=False, memory_key=memory_key,
            eval_kw={"use_mhn_classify": True, "mhn_beta": beta},
        ))

    runs.extend([
        _run("S4-01", 4, "t1_s4_proto_cosine_refine", fit=False, memory_key="prototype",
             eval_kw={**_REFINE_EVAL, "use_mhn_classify": False}),
        _run("S4-02", 4, "t1_s4_proto_mhn_refine", fit=False, memory_key="prototype",
             eval_kw={**_REFINE_EVAL, "use_mhn_classify": True}),
        _run("S4-03", 4, "t1_s4_exem_cosine_refine", fit=False, memory_key="exemplar",
             eval_kw={**_REFINE_EVAL, "use_mhn_classify": False}),
        _run("S4-04", 4, "t1_s4_exem_mhn_refine", fit=False, memory_key="exemplar",
             eval_kw={**_REFINE_EVAL, "use_mhn_classify": True}),
    ])

    runs.append({
        "id": "P0-01",
        "section": 0,
        "feature_pipeline": FEATURE_PIPELINE,
        "run_dir": "t1_mlp_classifier",
        "fit": True,
        "run_kind": "mlp_classifier",
        "fit_cfg": _mlp_classifier_cfg,
    })

    return runs


RUNS: list[dict[str, Any]] = _build_runs()

MEANINGFUL_STAGES: dict[str, list[str]] = {
    "P1_1_build": ["P1-01", "P1-02", "P1-03", "P1-04", "P1-05", "P1-06", "P1-07", "P1-08"],
    "P1_2_online": ["P1-09", "P1-10", "P1-11", "P1-all"],
    "P1_3_gates": ["P1-09", "P1-12", "P1-13", "P1-14", "P1-15", "P1-16", "P1-17", "P1-18"],
    "P1_gating_ablation": list(GATING_ABLATION_RUN_IDS),
    "S3_mhn_classify_proto": ["S3-01", "S3-02"],
    "S3_mhn_classify_exem": ["S3-03", "S3-04"],
    "S3_mhn_beta": [
        "S3-01", "S3-02", "S3-03", "S3-04",
        "S3-05", "S3-06", "S3-07", "S3-08", "S3-09", "S3-10",
    ],
    "S4_refine_proto": ["S4-01", "S4-02"],
    "S4_refine_exem": ["S4-03", "S4-04"],
    "P0_mlp": ["P0-01"],
    "E1_init": ["E2-02", "E2-04", "E2-05"],
    "E1_build": ["E2-01", "E2-02", "E2-03"],
    "E2_update": ["E2-06", "E2-07", "E2-08", "E2-09"],
}

_TRACK_STAGES = {
    "proto": ("P1_1_build", "P1_2_online", "P1_3_gates", "S3_mhn_classify_proto", "S4_refine_proto"),
    "exem": ("E1_init", "E1_build", "E2_update", "S3_mhn_classify_exem", "S4_refine_exem"),
}


def meaningful_run_ids(*stages: str, track: str | None = None) -> list[str]:
    if not stages:
        stages = _TRACK_STAGES.get(track or "", tuple(MEANINGFUL_STAGES))
    out: list[str] = []
    for stage in stages:
        if stage not in MEANINGFUL_STAGES:
            raise KeyError(f"Unknown stage {stage!r}; choose from {list(MEANINGFUL_STAGES)}")
        out.extend(MEANINGFUL_STAGES[stage])
    return list(dict.fromkeys(out))


def memory_source_for_run(run: dict[str, Any], repo: Path) -> str:
    if run.get("fit"):
        return str(run["run_dir"])
    if "memory_run_dir" in run:
        return str(run["memory_run_dir"])
    return CHAMPION[run.get("memory_key", "prototype")]


def get_run(run_id: str) -> dict[str, Any]:
    for run in RUNS:
        if run["id"] == run_id:
            return deepcopy(run)
    raise KeyError(f"Unknown run_id {run_id!r}")


def build_eval_cfg(repo: Path, run: dict[str, Any], memory_path: Path | None) -> dict[str, Any]:
    return run["eval_cfg"](repo, memory_path)


def iter_runs(section: int | None = None, run_ids: list[str] | None = None):
    for run in RUNS:
        if section is not None and run["section"] != section:
            continue
        if run_ids is not None and run["id"] not in run_ids:
            continue
        yield run


def resolve_paths(repo: Path, run: dict[str, Any]):
    run_dir = (repo / EXP_ROOT / run["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    mem_path = run_dir / "memory.pt"
    results_dir = (run_dir / "eval" / "eval").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, mem_path, results_dir
