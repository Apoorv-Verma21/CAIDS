import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.config import ensure_dirs, load_config
from cagids.data import chronological_split
from cagids.graphs import build_windowed_graphs, save_meta

LOG_FEATURES = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
                "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"]


def main():
    parser = argparse.ArgumentParser(description="Build time-windowed graphs and causal host context")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--split", choices=["chronological", "blocked", "random"], default=None)
    parser.add_argument("--no-structural", action="store_true")
    parser.add_argument("--no-behavioural", action="store_true")
    parser.add_argument("--no-role", action="store_true")
    parser.add_argument("--with-time", action="store_true")
    parser.add_argument("--window-seconds", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    overrides = {"context": {}, "split": {}, "data": {}}
    if args.no_structural:
        overrides["context"]["use_structural"] = False
    if args.no_behavioural:
        overrides["context"]["use_behavioural"] = False
    if args.no_role:
        overrides["context"]["use_role"] = False
    if args.with_time:
        overrides["context"]["use_time"] = True
    if args.split:
        overrides["split"]["mode"] = args.split
    if args.window_seconds is not None:
        overrides["data"]["window_seconds"] = args.window_seconds

    cfg = load_config(args.config, overrides=overrides, project_root=args.root)
    ensure_dirs(cfg)

    work_dir = cfg.get_path("work_dir")
    out_dir = work_dir / (args.tag or "graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(work_dir / "prepare_meta.json", "r", encoding="utf-8") as handle:
        prepare_meta = json.load(handle)

    print("loading flows ...", flush=True)
    df = pd.read_parquet(work_dir / "flows.parquet")

    if args.window_seconds is not None and args.window_seconds != prepare_meta["window_seconds"]:
        from cagids.data import assign_windows
        df = assign_windows(df.drop(columns=["window_id"]), args.window_seconds)
        df = df.sort_values("start_ms", kind="mergesort").reset_index(drop=True)

    print("building windowed graphs ...", flush=True)
    graphs, row_index, context_names = build_windowed_graphs(df, cfg)
    print("windows kept: {}".format(graphs.n_windows))
    print("edges: {}".format(len(graphs.edge_src)))
    print("context features: {} ({})".format(len(context_names), ", ".join(context_names)))

    feature_cols = prepare_meta["feature_cols"]
    edge_features = df[feature_cols].values.astype("float32")[row_index]
    labels = df["target"].values.astype("int64")[row_index]

    for name in LOG_FEATURES:
        if name in feature_cols:
            idx = feature_cols.index(name)
            edge_features[:, idx] = np.log1p(np.clip(edge_features[:, idx], 0, None))

    assignment = chronological_split(graphs.window_ids, cfg)
    splits = {"train": [], "val": [], "test": []}
    for i, window_id in enumerate(graphs.window_ids):
        splits[assignment[window_id]].append(i)

    if cfg.split.get("mode") == "random":
        rng = np.random.default_rng(int(cfg.train.get("seed", 42)))
        order = rng.permutation(graphs.n_windows)
        n_tr = int(graphs.n_windows * cfg.split["train"])
        n_va = int(graphs.n_windows * cfg.split["val"])
        splits = {
            "train": sorted(order[:n_tr].tolist()),
            "val": sorted(order[n_tr:n_tr + n_va].tolist()),
            "test": sorted(order[n_tr + n_va:].tolist()),
        }

    train_edge_mask = np.zeros(len(edge_features), dtype=bool)
    train_node_mask = np.zeros(len(graphs.context), dtype=bool)
    for w in splits["train"]:
        train_edge_mask[graphs.edge_offsets[w]:graphs.edge_offsets[w + 1]] = True
        train_node_mask[graphs.node_offsets[w]:graphs.node_offsets[w + 1]] = True

    if train_edge_mask.sum() == 0:
        raise ValueError("Training split is empty. Check split fractions in the config.")

    def fit_apply(matrix, mask):
        mean = matrix[mask].mean(axis=0, keepdims=True)
        std = matrix[mask].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return ((matrix - mean) / std).astype("float32")

    edge_features = fit_apply(edge_features, train_edge_mask)
    graphs.context = fit_apply(graphs.context, train_node_mask)

    graphs.save(out_dir)
    np.save(out_dir / "edge_features.npy", edge_features)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "row_index.npy", row_index)
    with open(out_dir / "splits.json", "w", encoding="utf-8") as handle:
        json.dump({k: list(map(int, v)) for k, v in splits.items()}, handle)

    class_names = prepare_meta["classes"]
    present = {}
    for split_name, windows in splits.items():
        rows = np.concatenate([
            np.arange(graphs.edge_offsets[w], graphs.edge_offsets[w + 1]) for w in windows
        ]) if windows else np.array([], dtype=int)
        counts = np.bincount(labels[rows], minlength=len(class_names))
        present[split_name] = {class_names[i]: int(counts[i]) for i in range(len(class_names))}

    save_meta(out_dir, {
        "context_names": context_names,
        "feature_cols": feature_cols,
        "classes": class_names,
        "n_windows": int(graphs.n_windows),
        "n_edges": int(len(edge_features)),
        "edge_dim": int(edge_features.shape[1]),
        "context_dim": int(graphs.context.shape[1]),
        "split_mode": cfg.split.get("mode"),
        "window_seconds": int(cfg.data.get("window_seconds", 60)),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_counts": present,
    })

    print("\nsplit windows: train={} val={} test={}".format(
        len(splits["train"]), len(splits["val"]), len(splits["test"])))
    print("\nclass counts per split:")
    print("{:<20} {:>12} {:>12} {:>12}".format("class", "train", "val", "test"))
    for name in class_names:
        print("{:<20} {:>12} {:>12} {:>12}".format(
            name[:20], present["train"][name], present["val"][name], present["test"][name]))

    missing = [n for n in class_names if present["test"][n] == 0]
    if missing:
        print("\nWARNING: missing from test split: {}".format(", ".join(missing)))
        print("Rerun this script with  --split blocked")

    print("\nsaved to {}".format(out_dir))


if __name__ == "__main__":
    main()
