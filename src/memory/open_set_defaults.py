""" Open-set UNKNOWN gates """

from __future__ import annotations

from typing import Any

# Stratified subsample cap for gate calibration (P1-18 / E2-09 / OWOD / T1 eval refit)
OPEN_SET_CALIB_MAX_RECORDS = 10_000

OPEN_SET_GATE_DEFAULTS: dict[str, Any] = {
    # UNKNOWN gates use cosine class-aware scores, label can still use MHN classify
    "open_set_gate_use_cosine_class": True,
    "use_global_threshold": True,
    "open_set_margin_mode": "class_aware",
    "use_proto_margin": True,
    "proto_margin_min": None,
    "open_set_threshold_mode": "known_only",
    "open_set_calib_method": "known_only",
    "open_set_calib_known_percentile": 5.0,
    "open_set_calib_known_margin_percentile": 5.0,
    "open_set_calib_split_file": "",
    # Subsample cap for collect_open_set_scores (None = use all calib records)
    "open_set_calib_max_records": OPEN_SET_CALIB_MAX_RECORDS,
    "open_set_calib_seed": 0,
}


def apply_open_set_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key, value in OPEN_SET_GATE_DEFAULTS.items():
        out.setdefault(key, value)
    return out
