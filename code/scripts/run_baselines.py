import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.baselines import RUNNERS, run_baseline
from cagids.config import ensure_dirs, load_config
from cagids.graphs import load_meta
from cagids.metrics import aggregate_seeds, format_metrics


def main():
    parser = argparse.ArgumentParser(description="Run flat-feature baselines on exactly the same flows and splits")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--graphs", default="graphs")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--drop-features", nargs="*", default=None,
                        help="flow features to remove before training")
    parser.add_argument("--save-val", action="store_true",
                        help="also save validation probabilities (for the prior correction check)")
    args = parser.parse_args()

    overrides = {"baselines": {}}
    if args.models:
        overrides["baselines"]["models"] = args.models

    cfg = load_config(args.config, overrides=overrides, project_root=args.root)
    ensure_dirs(cfg)

    graph_dir = cfg.get_path("work_dir") / args.graphs
    meta = load_meta(graph_dir)
    edge_features = np.load(graph_dir / "edge_features.npy")
    labels = np.load(graph_dir / "labels.npy")
    edge_split = np.load(graph_dir / "edge_split.npy")

    dropped = []
    if args.drop_features:
        missing = [c for c in args.drop_features if c not in meta["feature_cols"]]
        if missing:
            raise ValueError("unknown flow features: {}".format(", ".join(missing)))
        dropped = list(args.drop_features)
        edge_features = edge_features[:, [i for i, c in enumerate(meta["feature_cols"]) if c not in dropped]]

    class_names = meta["classes"]
    train_rows = np.flatnonzero(edge_split == 0)
    val_rows = np.flatnonzero(edge_split == 1)
    test_rows = np.flatnonzero(edge_split == 2)
    print("train flows: {}".format(len(train_rows)))
    print("test flows : {}".format(len(test_rows)))

    seeds = args.seeds if args.seeds else cfg.train.get("seeds", [cfg.train.get("seed", 42)])
    results_dir = cfg.get_path("results_dir")
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.tag is None else "_" + args.tag

    for name in cfg.baselines.get("models", list(RUNNERS)):
        if name not in RUNNERS:
            print("unknown baseline {}, skipped".format(name))
            continue
        runs = []
        for seed in seeds:
            metrics, probs, val_probs = run_baseline(
                name, edge_features[train_rows], labels[train_rows],
                edge_features[test_rows], labels[test_rows], class_names, cfg, int(seed),
                x_val=edge_features[val_rows] if args.save_val else None,
            )
            metrics["seed"] = int(seed)
            runs.append(metrics)
            print(format_metrics("baseline: {} (seed {})".format(name, seed), metrics))
            saved = dict(
                edge_ids=test_rows,
                y_true=labels[test_rows].astype("int16"),
                y_pred=probs.argmax(axis=1).astype("int16"),
                y_prob=probs.astype("float16"),
            )
            if val_probs is not None:
                saved.update(val_edge_ids=val_rows, val_y_true=labels[val_rows].astype("int16"),
                             val_y_prob=val_probs.astype("float32"), test_y_prob32=probs.astype("float32"))
            np.savez_compressed(pred_dir / "{}{}_seed{}.npz".format(name, suffix, seed), **saved)
            if metrics["macro_f1"] > 0.99:
                print("WARNING: macro F1 above 0.99 usually means a leaky feature.")

        summary = {
            "experiment": name,
            "tag": name + suffix,
            "graphs": str(graph_dir),
            "graphs_tag": args.graphs,
            "split_mode": meta.get("split_mode"),
            "window_seconds": meta.get("window_seconds"),
            "dropped_features": dropped,
            "runs": runs,
            "aggregate": aggregate_seeds(runs),
        }
        out_path = results_dir / "baseline_{}{}.json".format(name, suffix)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        agg = summary["aggregate"]
        print("\n{} across {} seed(s): macro F1 {:.4f} +/- {:.4f}".format(
            name, agg["n_seeds"], agg["macro_f1"]["mean"], agg["macro_f1"]["std"]))
        print("saved {}".format(out_path))


if __name__ == "__main__":
    main()
