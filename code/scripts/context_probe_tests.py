"""Why does endpoint context lower the macro-F1 of a random forest?

Two explanations are tested with the same forest as the context probes in
diagnostics.py (200 trees, balanced class weights, 400,000-flow subsample,
seeds 1 to 3):

1. dilution: the 34 context columns take split candidates away from the flow
   features. If so, the same number of uninformative columns (the context
   block with its rows shuffled) should hurt about as much as the real context.
2. shift: the forest uses the context as a stand-in for how particular hosts
   behaved in the training period, which does not carry over to the later test
   period. If so, the real context should hurt under the chronological split
   but not under the random flow split, and the shuffled context should not
   hurt.

The script also reports the share of the forest's impurity importance that
falls on context columns, and probes without the behavioural group (C2) and
with C2 alone.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.baselines import subsample_keep_attacks
from cagids.config import ensure_dirs, load_config
from cagids.context import BEHAVIOURAL_NAMES
from cagids.graphs import WindowedGraphs, load_meta
from cagids.metrics import compute_metrics

SEEDS = (1, 2, 3)


def endpoint_context(graphs, context_raw):
    src = graphs.node_offsets[graphs.edge_window] + graphs.edge_src
    dst = graphs.node_offsets[graphs.edge_window] + graphs.edge_dst
    return np.concatenate([context_raw[src], context_raw[dst]], axis=1)


def run_forest(x, labels, edge_split, class_names, limit, n_context):
    tr, te = edge_split == 0, edge_split == 2
    runs, shares = [], []
    for seed in SEEDS:
        x_sub, y_sub = subsample_keep_attacks(x[tr], labels[tr], limit, seed)
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=seed)
        rf.fit(x_sub, y_sub)
        pred = rf.classes_[rf.predict_proba(x[te]).argmax(axis=1)]
        m = compute_metrics(labels[te], pred, class_names)
        runs.append({k: m[k] for k in ("macro_f1", "macro_f1_attacks", "binary_f1", "false_positive_rate")})
        runs[-1]["per_class_f1"] = {c: m["per_class"][c]["f1"] for c in class_names}
        if n_context:
            shares.append(float(rf.feature_importances_[-n_context:].sum()))
    out = {}
    for key in ("macro_f1", "macro_f1_attacks", "binary_f1", "false_positive_rate"):
        vals = [r[key] for r in runs]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    out["per_class_f1"] = {c: float(np.mean([r["per_class_f1"][c] for r in runs])) for c in class_names}
    if shares:
        out["context_importance_share"] = {"mean": float(np.mean(shares)), "std": float(np.std(shares, ddof=1))}
    out["n_features"] = int(x.shape[1])
    return out


def main():
    parser = argparse.ArgumentParser(description="Tests of why endpoint context hurts a flat classifier")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--splits", nargs="*", default=["graphs:chronological", "graphs_random_flow:random_flow"],
                        help="graphs:name pairs; the extra probes run only for the chronological split")
    parser.add_argument("--out", default="context_probe_tests.json")
    args = parser.parse_args()

    cfg = load_config(args.config, project_root=args.root)
    ensure_dirs(cfg)
    work_dir = cfg.get_path("work_dir")
    limit = cfg.baselines.get("subsample_train", 400000)
    results = {}

    for graphs_tag, split_name in (s.split(":") for s in args.splits):
        graph_dir = work_dir / graphs_tag
        meta = load_meta(graph_dir)
        class_names = meta["classes"]
        graphs = WindowedGraphs.load(graph_dir)
        labels = np.load(graph_dir / "labels.npy")
        edge_split = np.load(graph_dir / "edge_split.npy")
        flow = np.load(graph_dir / "edge_features.npy")
        ctx = endpoint_context(graphs, np.load(graph_dir / "context_raw.npy"))

        names = meta["context_names"]
        c2 = np.array([n in BEHAVIOURAL_NAMES for n in names] * 2)
        rng = np.random.default_rng(0)
        shuffled = ctx[rng.permutation(len(ctx))]

        combos = [("flow_only", flow, 0), ("flow_plus_context", np.concatenate([flow, ctx], axis=1), ctx.shape[1])]
        if split_name == "chronological":
            combos += [
                ("flow_plus_shuffled_context", np.concatenate([flow, shuffled], axis=1), ctx.shape[1]),
                ("flow_plus_context_without_c2", np.concatenate([flow, ctx[:, ~c2]], axis=1), int((~c2).sum())),
                ("flow_plus_c2_only", np.concatenate([flow, ctx[:, c2]], axis=1), int(c2.sum())),
            ]
        for name, matrix, n_context in combos:
            t0 = time.perf_counter()
            res = run_forest(matrix, labels, edge_split, class_names, limit, n_context)
            results["{}/{}".format(split_name, name)] = res
            print("{:<14} {:<30} macro F1 {:.4f} +/- {:.4f}  attack {:.4f}  ({:.0f}s)".format(
                split_name, name, res["macro_f1"]["mean"], res["macro_f1"]["std"],
                res["macro_f1_attacks"]["mean"], time.perf_counter() - t0), flush=True)
            if "context_importance_share" in res:
                print("    share of impurity importance on context columns: {:.3f}".format(
                    res["context_importance_share"]["mean"]), flush=True)

    out_path = cfg.get_path("results_dir") / args.out
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("\nsaved {}".format(out_path))


if __name__ == "__main__":
    main()
