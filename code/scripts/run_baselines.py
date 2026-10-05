import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.baselines import run_baselines
from cagids.config import ensure_dirs, load_config
from cagids.graphs import WindowedGraphs, load_meta
from cagids.metrics import format_metrics


def rows_for(graphs, windows):
    if not windows:
        return np.array([], dtype=int)
    return np.concatenate([
        np.arange(graphs.edge_offsets[w], graphs.edge_offsets[w + 1]) for w in windows
    ])


def main():
    parser = argparse.ArgumentParser(description="Run flat-feature baselines on the same splits")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--graphs", default="graphs")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    overrides = {"baselines": {}}
    if args.models:
        overrides["baselines"]["models"] = args.models

    cfg = load_config(args.config, overrides=overrides, project_root=args.root)
    ensure_dirs(cfg)

    graph_dir = cfg.get_path("work_dir") / args.graphs
    meta = load_meta(graph_dir)
    graphs = WindowedGraphs.load(graph_dir)
    edge_features = np.load(graph_dir / "edge_features.npy")
    labels = np.load(graph_dir / "labels.npy")
    with open(graph_dir / "splits.json", "r", encoding="utf-8") as handle:
        splits = json.load(handle)

    class_names = meta["classes"]
    train_rows = rows_for(graphs, splits["train"])
    test_rows = rows_for(graphs, splits["test"])

    print("train flows: {}".format(len(train_rows)))
    print("test flows : {}".format(len(test_rows)))

    results = run_baselines(
        edge_features[train_rows], labels[train_rows],
        edge_features[test_rows], labels[test_rows],
        class_names, cfg,
    )

    for name, metrics in results.items():
        print(format_metrics("baseline: {}".format(name), metrics))

    out_path = cfg.get_path("results_dir") / "baselines.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({
            "graphs": str(graph_dir),
            "split_mode": meta.get("split_mode"),
            "results": results,
        }, handle, indent=2)

    high = [n for n, m in results.items() if m["macro_f1"] > 0.99]
    if high:
        print("\nWARNING: {} reached macro F1 above 0.99.".format(", ".join(high)))
        print("This usually means a leaky feature. Check data.leaky_features in the config.")

    print("\nsaved {}".format(out_path))


if __name__ == "__main__":
    main()
