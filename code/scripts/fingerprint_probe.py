"""Probe of the retained fields that could fingerprint a machine (TCP window sizes and DNS_QUERY_ID).

1. label probe: one depth-6 tree on all three fields, trained and scored exactly like the
   single-feature probes of diagnostics.py
2. host probe: a deeper tree on the same fields that predicts the source address of a flow,
   compared with always predicting the most common training source

    python scripts/fingerprint_probe.py --root D:/CAIDS_work
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.baselines import subsample_keep_attacks
from cagids.config import load_config
from cagids.graphs import load_meta
from cagids.metrics import compute_metrics

FINGERPRINT_FIELDS = ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_QUERY_ID"]


def main():
    parser = argparse.ArgumentParser(description="Probe of possible machine fingerprints")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--graphs", default="graphs")
    parser.add_argument("--probe-benign", type=int, default=300000)
    parser.add_argument("--host-depth", type=int, default=None)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    cfg = load_config(args.config, project_root=args.root)
    work_dir = cfg.get_path("work_dir")
    graph_dir = work_dir / args.graphs
    results_dir = cfg.get_path("results_dir")

    meta = load_meta(graph_dir)
    class_names = meta["classes"]
    row_index = np.load(graph_dir / "row_index.npy")
    labels = np.load(graph_dir / "labels.npy")
    edge_split = np.load(graph_dir / "edge_split.npy")
    df = pd.read_parquet(work_dir / "flows.parquet", columns=FINGERPRINT_FIELDS + ["src_ip", "target"])
    flows = df.iloc[row_index].reset_index(drop=True)
    assert np.array_equal(flows["target"].values.astype("int64"), labels), "label alignment failed"

    tr, te = edge_split == 0, edge_split == 2
    x_all = flows[FINGERPRINT_FIELDS].to_numpy(dtype="float64")
    out = {"fields": FINGERPRINT_FIELDS, "graphs": args.graphs}

    # 1. label probe, same sample and tree as the single-feature probes
    rng_idx = np.flatnonzero(tr)
    sub_idx, _ = subsample_keep_attacks(rng_idx, labels[rng_idx],
                                        args.probe_benign + int((labels[rng_idx] > 0).sum()), 0)
    tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0)
    tree.fit(x_all[sub_idx], labels[sub_idx])
    m = compute_metrics(labels[te], tree.predict(x_all[te]), class_names)
    out["label_probe"] = {k: float(m[k]) for k in ("macro_f1", "macro_f1_attacks", "binary_f1", "false_positive_rate")}

    # 2. host probe: which machine sent the flow
    src = flows["src_ip"].values
    host_tree = DecisionTreeClassifier(max_depth=args.host_depth, random_state=0)
    host_tree.fit(x_all[sub_idx], src[sub_idx])
    pred = host_tree.predict(x_all[te])
    majority = pd.Series(src[sub_idx]).value_counts().idxmax()
    out["host_probe"] = {
        "depth": args.host_depth,
        "accuracy": float(np.mean(pred == src[te])),
        "balanced_accuracy": float(balanced_accuracy_score(src[te], pred)),
        "majority_accuracy": float(np.mean(src[te] == majority)),
        "n_sources_test": int(len(np.unique(src[te]))),
    }
    print(json.dumps(out, indent=2))
    out_path = results_dir / "fingerprint_probe_{}.json".format(args.graphs)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("saved {}".format(out_path))


if __name__ == "__main__":
    main()
