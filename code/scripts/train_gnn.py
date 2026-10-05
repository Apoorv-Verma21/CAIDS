import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.config import ensure_dirs, load_config
from cagids.graphs import WindowedGraphs, load_meta
from cagids.metrics import aggregate_seeds, format_metrics
from cagids.models import build_model
from cagids.training import set_seed, train_model


EXPERIMENTS = {
    "egraphsage": {"model": {"name": "egraphsage"}},
    "centrality_egraphsage": {"model": {"name": "centrality_egraphsage"}},
    "cagids": {"model": {"name": "cagids"}},
    "cagids_concat": {"model": {"name": "cagids_concat"}},
}


def main():
    parser = argparse.ArgumentParser(description="Train a graph model and evaluate on the test split")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--experiment", default="cagids", choices=sorted(EXPERIMENTS))
    parser.add_argument("--graphs", default="graphs")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    overrides = dict(EXPERIMENTS[args.experiment])
    overrides.setdefault("train", {})
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.device is not None:
        overrides["train"]["device"] = args.device

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
    structural_dim = sum(1 for n in meta["context_names"] if n in
                         ["in_degree", "out_degree", "unique_peers", "pagerank", "clustering"])

    print("experiment      : {}".format(args.experiment))
    print("graphs          : {}".format(graph_dir))
    print("windows         : train={} val={} test={}".format(
        len(splits["train"]), len(splits["val"]), len(splits["test"])))
    print("edge features   : {}".format(edge_features.shape[1]))
    print("context features: {}".format(graphs.context.shape[1]))
    print("classes         : {}".format(len(class_names)))

    seeds = args.seeds if args.seeds else cfg.train.get("seeds", [cfg.train.get("seed", 42)])
    runs = []
    for seed in seeds:
        print("\n--- seed {} ---".format(seed), flush=True)
        set_seed(int(seed))
        model_cfg = dict(cfg["model"])
        model_cfg["structural_dim"] = structural_dim
        run_cfg = type(cfg)({**cfg, "model": model_cfg})

        model = build_model(
            edge_dim=edge_features.shape[1],
            context_dim=graphs.context.shape[1],
            n_classes=len(class_names),
            cfg=run_cfg,
        )
        _, metrics = train_model(
            model, graphs, splits, edge_features, labels, class_names, run_cfg
        )
        metrics["seed"] = int(seed)
        runs.append(metrics)
        print(format_metrics("{} (seed {})".format(args.experiment, seed), metrics))

    summary = {
        "experiment": args.experiment,
        "graphs": str(graph_dir),
        "split_mode": meta.get("split_mode"),
        "window_seconds": meta.get("window_seconds"),
        "runs": runs,
        "aggregate": aggregate_seeds(runs),
    }

    tag = args.tag or args.experiment
    results_dir = cfg.get_path("results_dir")
    out_path = results_dir / "gnn_{}.json".format(tag)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if len(runs) > 1:
        agg = summary["aggregate"]
        print("\nacross {} seeds: macro F1 {:.4f} +/- {:.4f}".format(
            agg["n_seeds"], agg["macro_f1"]["mean"], agg["macro_f1"]["std"]))

    print("\nsaved {}".format(out_path))


if __name__ == "__main__":
    main()
