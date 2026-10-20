"""Dataset and shortcut diagnostics.

1. dataset statistics: flows, windows, hosts, capture sessions, classes per split
2. temporal class distribution (input for the timeline figure)
3. host overlap between the training and test periods
4. identity probes: how far host identity or service port alone predicts the label
5. single-feature probes: a depth-limited tree on one feature at a time, including
   the columns excluded as leaky
6. context probes: random forest on the endpoints' context, with and without the
   flow features (separates "is the context informative" from "is the GNN needed")
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.baselines import subsample_keep_attacks
from cagids.config import ensure_dirs, load_config
from cagids.graphs import WindowedGraphs, load_meta
from cagids.metrics import compute_metrics

SESSION_GAP_MS = 3_600_000


def scores(y_true, y_pred, class_names):
    m = compute_metrics(y_true, y_pred, class_names)
    return {k: m[k] for k in ("macro_f1", "macro_f1_attacks", "weighted_f1", "binary_f1",
                              "detection_rate", "false_positive_rate") if k in m}


def majority_lookup(train_keys, train_y, test_keys, default=0):
    frame = pd.DataFrame({"k": train_keys, "y": train_y})
    table = frame.groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
    pred = pd.Series(test_keys).map(table).fillna(default).astype(int).values
    return pred


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics and shortcut diagnostics")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--graphs", default="graphs")
    parser.add_argument("--probe-benign", type=int, default=300000)
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    cfg = load_config(args.config, project_root=args.root)
    ensure_dirs(cfg)
    work_dir = cfg.get_path("work_dir")
    graph_dir = work_dir / args.graphs
    results_dir = cfg.get_path("results_dir")

    meta = load_meta(graph_dir)
    with open(work_dir / "prepare_meta.json", "r", encoding="utf-8") as handle:
        prepare_meta = json.load(handle)
    class_names = meta["classes"]
    graphs = WindowedGraphs.load(graph_dir)
    row_index = np.load(graph_dir / "row_index.npy")
    labels = np.load(graph_dir / "labels.npy")
    edge_split = np.load(graph_dir / "edge_split.npy")
    context_raw = np.load(graph_dir / "context_raw.npy")

    print("loading flows ...", flush=True)
    df = pd.read_parquet(work_dir / "flows.parquet")
    flows = df.iloc[row_index].reset_index(drop=True)
    assert np.array_equal(flows["target"].values.astype("int64"), labels), "label alignment failed"

    out = {"graphs": args.graphs, "split_mode": meta.get("split_mode"),
           "window_seconds": meta.get("window_seconds")}

    # 1. dataset statistics ---------------------------------------------------
    start = df["start_ms"].values
    gaps = np.flatnonzero(np.diff(start) > SESSION_GAP_MS)
    bounds = np.concatenate([[0], gaps + 1, [len(df)]])
    sessions = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        part = df.iloc[a:b]
        sessions.append({
            "start_utc": str(pd.to_datetime(part["start_ms"].iloc[0], unit="ms", utc=True)),
            "end_utc": str(pd.to_datetime(part["start_ms"].iloc[-1], unit="ms", utc=True)),
            "hours": float((part["start_ms"].iloc[-1] - part["start_ms"].iloc[0]) / 3.6e6),
            "flows": int(len(part)),
            "attack_flows": int(part["label"].sum()),
        })

    flows_per_window = np.diff(graphs.edge_offsets)
    hosts_per_window = np.diff(graphs.node_offsets)
    attack_src = flows.loc[flows["label"] == 1, "src_ip"].value_counts()
    attack_dst = flows.loc[flows["label"] == 1, "dst_ip"].value_counts()
    benign_src = flows.loc[flows["label"] == 0, "src_ip"].value_counts()
    out["dataset"] = {
        "flows_raw": int(prepare_meta.get("n_flows", len(df))),
        "flows_in_graphs": int(len(row_index)),
        "flows_dropped_small_windows": int(len(df) - len(row_index)),
        "attack_rate": float(flows["label"].mean()),
        "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(class_names)},
        "n_feature_columns": len(meta["feature_cols"]),
        "feature_columns": meta["feature_cols"],
        "n_context_features": len(meta["context_names"]),
        "sessions": sessions,
        "windows": int(graphs.n_windows),
        "flows_per_window": {"median": float(np.median(flows_per_window)), "mean": float(flows_per_window.mean()),
                             "max": int(flows_per_window.max()), "p95": float(np.percentile(flows_per_window, 95))},
        "hosts_per_window": {"median": float(np.median(hosts_per_window)), "mean": float(hosts_per_window.mean()),
                             "max": int(hosts_per_window.max())},
        "unique_hosts": int(pd.unique(pd.concat([flows["src_ip"], flows["dst_ip"]])).size),
        "unique_attack_sources": int(len(attack_src)),
        "top_attack_sources": {k: int(v) for k, v in attack_src.head(10).items()},
        "top_attack_destinations": {k: int(v) for k, v in attack_dst.head(10).items()},
        "attack_share_top4_sources": float(attack_src.head(4).sum() / max(attack_src.sum(), 1)),
        "attack_sources_also_benign_sources": int(len(set(attack_src.index) & set(benign_src.index))),
    }

    split_info = {}
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        sel = edge_split == code
        part = flows.loc[sel]
        split_info[name] = {
            "flows": int(sel.sum()),
            "attack_flows": int(part["label"].sum()),
            "start_utc": str(pd.to_datetime(part["start_ms"].min(), unit="ms", utc=True)),
            "end_utc": str(pd.to_datetime(part["start_ms"].max(), unit="ms", utc=True)),
            "windows": int(len(np.unique(graphs.edge_window[sel]))),
            "class_counts": {c: int((labels[sel] == i).sum()) for i, c in enumerate(class_names)},
        }
    out["splits"] = split_info

    # 2. temporal class distribution -----------------------------------------
    bin_ms = 10 * 60 * 1000
    tl = pd.DataFrame({
        "bin": (flows["start_ms"].values // bin_ms).astype("int64"),
        "cls": labels,
        "split": edge_split,
    })
    counts = tl.groupby(["bin", "cls"]).size().unstack(fill_value=0)
    counts.columns = [class_names[c] for c in counts.columns]
    split_of_bin = tl.groupby("bin")["split"].agg(lambda s: int(s.mode().iloc[0]))
    counts["split"] = split_of_bin
    counts.index = counts.index * bin_ms
    counts.index.name = "bin_start_ms"
    counts.to_csv(results_dir / "timeline_{}.csv".format(args.graphs))

    # 3. host overlap -----------------------------------------------------------
    tr, te = edge_split == 0, edge_split == 2
    hosts_tr = set(flows.loc[tr, "src_ip"]) | set(flows.loc[tr, "dst_ip"])
    hosts_te = set(flows.loc[te, "src_ip"]) | set(flows.loc[te, "dst_ip"])
    src_te = flows.loc[te, "src_ip"].values
    dst_te = flows.loc[te, "dst_ip"].values
    seen_src = np.array([s in hosts_tr for s in src_te])
    seen_dst = np.array([d in hosts_tr for d in dst_te])
    attack_src_tr = set(flows.loc[tr & (flows["label"].values == 1), "src_ip"])
    te_attack = te & (flows["label"].values == 1)
    att_src_te = flows.loc[te_attack, "src_ip"].values
    out["host_overlap"] = {
        "hosts_train": len(hosts_tr),
        "hosts_test": len(hosts_te),
        "test_hosts_seen_in_train": float(len(hosts_te & hosts_tr) / max(len(hosts_te), 1)),
        "test_flows_src_seen": float(seen_src.mean()),
        "test_flows_both_seen": float((seen_src & seen_dst).mean()),
        "attack_sources_train": len(attack_src_tr),
        "attack_sources_test": int(len(set(att_src_te))),
        "test_attack_flows_from_train_attack_sources": float(np.mean([s in attack_src_tr for s in att_src_te]))
        if len(att_src_te) else float("nan"),
    }

    # 4. identity probes ---------------------------------------------------------
    y_tr, y_te = labels[tr], labels[te]
    probes = {}
    keys = {
        "src_ip": flows["src_ip"].values,
        "dst_ip": flows["dst_ip"].values,
        "dst_port": flows["dst_port"].values,
        "src_ip+dst_ip": (flows["src_ip"] + ">" + flows["dst_ip"]).values,
        "src_ip+dst_ip+dst_port": (flows["src_ip"] + ">" + flows["dst_ip"] + ":" + flows["dst_port"].astype(str)).values,
    }
    for name, key in keys.items():
        pred = majority_lookup(key[tr], y_tr, key[te])
        probes[name] = scores(y_te, pred, class_names)
        print("identity probe {:<24} macro F1 {:.4f}  binary F1 {:.4f}".format(
            name, probes[name]["macro_f1"], probes[name]["binary_f1"]), flush=True)
    out["identity_probes"] = probes

    # 5. single-feature probes ----------------------------------------------------
    feature_cols = list(meta["feature_cols"])
    probe_cols = [c for c in flows.columns if c.startswith("probe__")]
    rng_idx = np.flatnonzero(tr)
    sub_x_idx, _ = subsample_keep_attacks(rng_idx, labels[rng_idx], args.probe_benign + int((labels[rng_idx] > 0).sum()), 0)
    single = {}
    for col in feature_cols + probe_cols:
        # the excluded per-second byte rates contain inf where the duration is zero
        x_all = flows[col].to_numpy(dtype="float64").reshape(-1, 1)
        x_all = np.clip(np.nan_to_num(x_all, nan=0.0, posinf=1e30, neginf=-1e30), -1e30, 1e30)
        tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0)
        tree.fit(x_all[sub_x_idx], labels[sub_x_idx])
        pred = tree.predict(x_all[te])
        single[col] = scores(y_te, pred, class_names)
    ranked = sorted(single.items(), key=lambda kv: -kv[1]["macro_f1"])
    out["single_feature_probes"] = dict(ranked)
    print("\ntop single-feature probes (macro F1 / binary F1):")
    for col, s in ranked[:12]:
        print("  {:<40} {:.4f}  {:.4f}".format(col, s["macro_f1"], s["binary_f1"]))
    for col in probe_cols:
        print("  excluded column {:<30} {:.4f}  {:.4f}".format(col, single[col]["macro_f1"], single[col]["binary_f1"]))

    # 6. context probes --------------------------------------------------------------
    if not args.skip_context:
        g_src = graphs.node_offsets[graphs.edge_window] + graphs.edge_src
        g_dst = graphs.node_offsets[graphs.edge_window] + graphs.edge_dst
        ctx_pair = np.concatenate([context_raw[g_src], context_raw[g_dst]], axis=1)
        edge_features = np.load(graph_dir / "edge_features.npy")
        combos = {
            "context_only_rf": ctx_pair,
            "flow_only_rf": edge_features,
            "flow_plus_context_rf": np.concatenate([edge_features, ctx_pair], axis=1),
        }
        context_results = {}
        for name, matrix in combos.items():
            t0 = time.perf_counter()
            runs = []
            for seed in (1, 2, 3):
                x_sub, y_sub = subsample_keep_attacks(matrix[tr], y_tr, cfg.baselines.get("subsample_train", 400000), seed)
                rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=seed)
                rf.fit(x_sub, y_sub)
                pred = rf.classes_[rf.predict_proba(matrix[te]).argmax(axis=1)]
                runs.append(scores(y_te, pred, class_names))
            context_results[name] = {
                k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs], ddof=1))}
                for k in runs[0]
            }
            print("{:<22} macro F1 {:.4f} +/- {:.4f}  ({:.0f}s)".format(
                name, context_results[name]["macro_f1"]["mean"], context_results[name]["macro_f1"]["std"],
                time.perf_counter() - t0), flush=True)
        out["context_probes"] = context_results

    out_path = results_dir / "diagnostics_{}.json".format(args.graphs)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)
    print("\nsaved {}".format(out_path))


if __name__ == "__main__":
    main()
