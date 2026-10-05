import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.config import ensure_dirs, load_config
from cagids.data import assign_windows, load_flows, prepare_frame


def main():
    parser = argparse.ArgumentParser(description="Clean the raw CSV and assign time windows")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--csv", default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--window-seconds", type=int, default=None)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    overrides = {"data": {}}
    if args.nrows is not None:
        overrides["data"]["nrows"] = args.nrows
    if args.window_seconds is not None:
        overrides["data"]["window_seconds"] = args.window_seconds

    cfg = load_config(args.config, overrides=overrides, project_root=args.root)
    ensure_dirs(cfg)

    csv_path = args.csv or str(cfg.get_path("raw_csv"))
    work_dir = cfg.get_path("work_dir")

    print("loading", csv_path, flush=True)
    raw = load_flows(csv_path, nrows=cfg.data.get("nrows"))
    print("raw rows: {}".format(len(raw)), flush=True)

    df, meta = prepare_frame(raw, cfg)
    del raw

    window_seconds = cfg.data.get("window_seconds", 60)
    df = assign_windows(df, window_seconds)

    classes = ["Benign"] + sorted(c for c in df["attack"].unique() if c != "Benign")
    class_to_index = {name: i for i, name in enumerate(classes)}
    df["target"] = df["attack"].map(class_to_index).astype("int16")

    out_path = work_dir / "flows.parquet"
    df.to_parquet(out_path, index=False)

    meta.update({
        "n_flows": int(len(df)),
        "window_seconds": int(window_seconds),
        "n_windows": int(df["window_id"].nunique()),
        "classes": classes,
        "attack_rate": float(df["label"].mean()),
        "source_csv": csv_path,
    })
    with open(work_dir / "prepare_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print("\nsaved {}".format(out_path))
    print("flows: {}".format(len(df)))
    print("windows: {}".format(meta["n_windows"]))
    print("features: {}".format(len(meta["feature_cols"])))
    print("classes: {}".format(", ".join(classes)))
    print("attack rate: {:.3f}%".format(100 * meta["attack_rate"]))


if __name__ == "__main__":
    main()
