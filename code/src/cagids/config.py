import os
from pathlib import Path

import yaml


DEFAULTS = {
    "paths": {
        "raw_csv": "data/NF-UNSW-NB15-v3.csv",
        "work_dir": "outputs/work",
        "results_dir": "outputs/results",
    },
    "data": {
        "nrows": None,
        "window_seconds": 60,
        "drop_leaky": True,
        "leaky_features": [
            "MIN_TTL",
            "MAX_TTL",
            "SRC_TO_DST_SECOND_BYTES",
            "DST_TO_SRC_SECOND_BYTES",
        ],
        "min_flows_per_window": 2,
        "max_windows": None,
    },
    "split": {
        "mode": "chronological",
        "train": 0.7,
        "val": 0.1,
        "test": 0.2,
        "blocks": 10,
    },
    "context": {
        "use_structural": True,
        "use_behavioural": True,
        "use_role": True,
        "use_time": False,
        "ewma_alpha": 0.3,
        "pagerank_iters": 30,
        "pagerank_damping": 0.85,
    },
    "model": {
        "name": "cagids",
        "hidden_dim": 128,
        "context_dim": 64,
        "layers": 2,
        "dropout": 0.2,
        "node_init": "context",
        "fusion": "gate",
    },
    "train": {
        "epochs": 30,
        "lr": 0.003,
        "weight_decay": 0.00001,
        "patience": 6,
        "seed": 42,
        "seeds": [42],
        "class_weighting": "inverse",
        "windows_per_batch": 8,
        "device": "auto",
    },
    "baselines": {
        "models": ["random_forest", "xgboost", "mlp"],
        "n_estimators": 200,
        "max_depth": None,
        "subsample_train": 400000,
    },
}


def _deep_merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def get_path(self, key):
        root = Path(self.get("project_root", "."))
        return root / self["paths"][key]


def load_config(path=None, overrides=None, project_root=None):
    user = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            user = yaml.safe_load(handle) or {}
    merged = _deep_merge(DEFAULTS, user)
    merged = _deep_merge(merged, overrides or {})
    merged["project_root"] = str(project_root or os.environ.get("CAGIDS_ROOT", "."))
    return Config(merged)


def ensure_dirs(cfg):
    for key in ("work_dir", "results_dir"):
        cfg.get_path(key).mkdir(parents=True, exist_ok=True)
