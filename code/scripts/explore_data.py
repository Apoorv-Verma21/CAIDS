import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.config import load_config
from cagids.data import assign_windows, chronological_split, describe_resolution, load_flows, prepare_frame


def main():
    parser = argparse.ArgumentParser(description="Inspect the raw dataset before running the pipeline")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--csv", default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    cfg = load_config(args.config, project_root=args.root)
    csv_path = args.csv or str(cfg.get_path("raw_csv"))

    print("reading:", csv_path)
    header = pd.read_csv(csv_path, nrows=5, low_memory=False)
    header.columns = [str(c).strip() for c in header.columns]

    print("\ncolumns found: {}".format(len(header.columns)))
    for col in header.columns:
        print("  " + col)

    summary, resolved = describe_resolution(header.columns)
    print("\n" + summary)

    nrows = args.nrows if args.nrows is not None else cfg.data.get("nrows")
    print("\nloading flows (nrows={}) ...".format(nrows))
    raw = load_flows(csv_path, nrows=nrows)
    df, meta = prepare_frame(raw, cfg)
    print("usable flows: {}".format(len(df)))
    print("feature columns kept: {}".format(len(meta["feature_cols"])))
    print("columns excluded from features: {}".format(", ".join(meta["dropped_cols"])))

    print("\nclass distribution:")
    counts = df["attack"].value_counts()
    for name, count in counts.items():
        print("  {:<20} {:>10}  ({:.3f}%)".format(name, count, 100 * count / len(df)))
    print("overall attack rate: {:.3f}%".format(100 * df["label"].mean()))

    span_ms = df["start_ms"].max() - df["start_ms"].min()
    print("\ntime span: {:.2f} hours".format(span_ms / 3600000.0))

    df = assign_windows(df, cfg.data.get("window_seconds", 60))
    n_windows = df["window_id"].nunique()
    print("windows at {}s: {}".format(cfg.data.get("window_seconds", 60), n_windows))
    print("median flows per window: {:.0f}".format(df.groupby("window_id").size().median()))

    assignment = chronological_split(df["window_id"].values, cfg)
    df["split"] = df["window_id"].map(assignment)

    print("\nchronological split check:")
    table = pd.crosstab(df["attack"], df["split"])
    for split in ("train", "val", "test"):
        if split not in table.columns:
            table[split] = 0
    table = table[["train", "val", "test"]]
    print(table.to_string())

    missing = [cls for cls in table.index if table.loc[cls, "test"] == 0]
    print()
    if missing:
        print("WARNING: these classes have no flows in the test split:")
        for cls in missing:
            print("  - {}".format(cls))
        print("\nFix: rerun later stages with  --split blocked")
        print("That assigns contiguous blocks to each split while keeping time order inside them.")
    else:
        print("All classes appear in the test split. Chronological split is usable.")

    numeric = df[meta["feature_cols"]]
    constant = [c for c in numeric.columns if numeric[c].std() < 1e-9]
    if constant:
        print("\nconstant features (consider removing): {}".format(", ".join(constant)))

    print("\nleakage watch: if a baseline reaches above 0.99 macro F1, inspect feature importances.")


if __name__ == "__main__":
    main()
