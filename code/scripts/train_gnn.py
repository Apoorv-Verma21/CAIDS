import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.baselines import subsample_keep_attacks
from cagids.config import ensure_dirs, load_config
from cagids.context import STRUCTURAL_NAMES
from cagids.graphs import WindowedGraphs, load_meta
from cagids.metrics import aggregate_seeds, format_metrics
from cagids.models import EXPERIMENT_MODELS, build_model
from cagids.training import set_seed, train_model, windows_for_split

UNUSED = 3  # split code for training flows left out of the loss


def subsample_training_flows(edge_split, labels, limit, seed):
    """Keep every training attack flow and a random set of benign training flows,
    up to `limit` in total, using the same function and seed as the flat
    baselines, so that a model trained this way sees the forest's training set.
    The other training flows stay in the graphs but leave the loss."""
    train_rows = np.flatnonzero(edge_split == 0)
    keep, _ = subsample_keep_attacks(train_rows, labels[train_rows], limit, seed)
    out = edge_split.copy()
    out[train_rows] = UNUSED
    out[keep] = 0
    return out


def training_node_mask(graphs, edge_split):
    """Nodes of the windows that contain training flows, the same nodes whose
    statistics standardise the context in build_graphs."""
    mask = np.zeros(len(graphs.context), dtype=bool)
    for w in windows_for_split(graphs, edge_split, "train"):
        mask[graphs.node_offsets[w]:graphs.node_offsets[w + 1]] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description="Train a graph model (or the flow-only MLP) and evaluate on the test split")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--experiment", default="cagids", choices=sorted(EXPERIMENT_MODELS))
    parser.add_argument("--graphs", default="graphs")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--loss", choices=["ce", "focal"], default=None)
    parser.add_argument("--class-weighting", choices=["inverse", "sqrt_inverse", "none"], default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--root", default=ROOT)
    # options for the robustness checks; the defaults keep the planned settings
    parser.add_argument("--train-subsample", type=int, default=None,
                        help="train on all attack flows plus benign flows up to this total, as the flat baselines do")
    parser.add_argument("--windows-per-batch", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--aggregator", choices=["mean", "max"], default=None)
    parser.add_argument("--direction-aware", action="store_true")
    parser.add_argument("--drop-features", nargs="*", default=None,
                        help="flow features to remove from the flow vector")
    parser.add_argument("--drop-context", nargs="*", default=None,
                        help="context features to remove from the node input")
    parser.add_argument("--clip-context", nargs="*", default=None,
                        help="context features to clip at their maximum over the training windows")
    parser.add_argument("--early-stop-exclude", nargs="*", default=None,
                        help="classes left out of the validation macro-F1 that early stopping monitors")
    args = parser.parse_args()

    overrides = {"model": {"name": args.experiment}, "train": {}}
    if args.layers is not None:
        overrides["model"]["layers"] = args.layers
    if args.aggregator is not None:
        overrides["model"]["aggregator"] = args.aggregator
    if args.direction_aware:
        overrides["model"]["direction_aware"] = True
    if args.windows_per_batch is not None:
        overrides["train"]["windows_per_batch"] = args.windows_per_batch
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.device is not None:
        overrides["train"]["device"] = args.device
    if args.loss is not None:
        overrides["train"]["loss"] = args.loss
    if args.class_weighting is not None:
        overrides["train"]["class_weighting"] = args.class_weighting
    if args.patience is not None:
        overrides["train"]["patience"] = args.patience
    if args.early_stop_exclude:
        overrides["train"]["early_stop_exclude"] = list(args.early_stop_exclude)

    cfg = load_config(args.config, overrides=overrides, project_root=args.root)
    ensure_dirs(cfg)

    graph_dir = cfg.get_path("work_dir") / args.graphs
    meta = load_meta(graph_dir)
    graphs = WindowedGraphs.load(graph_dir)
    edge_features = np.load(graph_dir / "edge_features.npy")
    labels = np.load(graph_dir / "labels.npy")
    edge_split = np.load(graph_dir / "edge_split.npy")

    dropped = []
    if args.drop_features:
        missing = [c for c in args.drop_features if c not in meta["feature_cols"]]
        if missing:
            raise ValueError("unknown flow features: {}".format(", ".join(missing)))
        dropped = list(args.drop_features)
        keep_cols = [i for i, c in enumerate(meta["feature_cols"]) if c not in dropped]
        edge_features = np.ascontiguousarray(edge_features[:, keep_cols])

    # The context is standardised, which preserves order, so clipping it at its
    # largest training value equals clipping the raw feature at that value.
    context_names = list(meta["context_names"])
    for name in args.clip_context or []:
        if name not in context_names:
            raise ValueError("unknown context feature: {}".format(name))
        j = context_names.index(name)
        cap = graphs.context[training_node_mask(graphs, edge_split), j].max()
        graphs.context[:, j] = np.minimum(graphs.context[:, j], cap)
    if args.drop_context:
        missing = [c for c in args.drop_context if c not in context_names]
        if missing:
            raise ValueError("unknown context features: {}".format(", ".join(missing)))
        keep_ctx = [j for j, c in enumerate(context_names) if c not in args.drop_context]
        graphs.context = np.ascontiguousarray(graphs.context[:, keep_ctx])
        context_names = [context_names[j] for j in keep_ctx]

    class_names = meta["classes"]
    structural_dim = sum(1 for n in context_names if n in STRUCTURAL_NAMES)
    if args.experiment == "centrality_egraphsage" and structural_dim == 0:
        raise ValueError("centrality_egraphsage needs structural context; these graphs were built without it")

    tag = args.tag or args.experiment
    print("experiment      : {}".format(args.experiment))
    print("graphs          : {} (split={}, window={}s)".format(graph_dir, meta.get("split_mode"), meta.get("window_seconds")))
    print("flows           : train={} val={} test={}".format(
        int((edge_split == 0).sum()), int((edge_split == 1).sum()), int((edge_split == 2).sum())))
    print("edge features   : {}".format(edge_features.shape[1]))
    print("context features: {}".format(graphs.context.shape[1]))
    print("classes         : {}".format(len(class_names)))

    seeds = args.seeds if args.seeds else cfg.train.get("seeds", [cfg.train.get("seed", 42)])
    results_dir = cfg.get_path("results_dir")
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

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
        run_split = edge_split
        if args.train_subsample:
            run_split = subsample_training_flows(edge_split, labels, args.train_subsample, int(seed))
            print("training flows after subsampling: {}".format(int((run_split == 0).sum())))
        _, metrics, predictions = train_model(
            model, graphs, run_split, edge_features, labels, class_names, run_cfg
        )
        metrics["seed"] = int(seed)
        runs.append(metrics)
        print(format_metrics("{} (seed {})".format(tag, seed), metrics))
        if metrics.get("inference"):
            print("inference: {:.2f} ms/window, {:.0f} flows/s".format(
                metrics["inference"]["ms_per_window_mean"], metrics["inference"]["flows_per_second"]))
        if not args.no_save_predictions:
            np.savez_compressed(pred_dir / "{}_seed{}.npz".format(tag, seed), **predictions)

    summary = {
        "experiment": args.experiment,
        "tag": tag,
        "graphs": str(graph_dir),
        "graphs_tag": args.graphs,
        "split_mode": meta.get("split_mode"),
        "window_seconds": meta.get("window_seconds"),
        "context_names": context_names,
        "clipped_context": list(args.clip_context or []),
        "loss": cfg.train.get("loss", "ce"),
        "class_weighting": cfg.train.get("class_weighting", "inverse"),
        "epochs": int(cfg.train.get("epochs", 30)),
        "patience": int(cfg.train.get("patience", 6)),
        "early_stop_exclude": list(args.early_stop_exclude or []),
        "windows_per_batch": int(cfg.train.get("windows_per_batch", 8)),
        "layers": int(cfg.model.get("layers", 2)),
        "aggregator": cfg.model.get("aggregator", "mean"),
        "direction_aware": bool(cfg.model.get("direction_aware", False)),
        "train_subsample": args.train_subsample,
        "dropped_features": dropped,
        "dropped_context": list(args.drop_context or []),
        "runs": runs,
        "aggregate": aggregate_seeds(runs),
    }

    out_path = results_dir / "gnn_{}.json".format(tag)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    agg = summary["aggregate"]
    print("\nacross {} seed(s): macro F1 {:.4f} +/- {:.4f}".format(
        agg["n_seeds"], agg["macro_f1"]["mean"], agg["macro_f1"]["std"]))
    print("saved {}".format(out_path))


if __name__ == "__main__":
    main()
