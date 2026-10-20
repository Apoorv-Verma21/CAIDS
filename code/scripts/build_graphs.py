import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.config import ensure_dirs, load_config
from cagids.data import assign_windows, chronological_split, random_flow_split
from cagids.graphs import build_windowed_graphs, save_meta

# Heavy-tailed, non-negative counts get log1p before standardisation:
# bytes, packets, durations, throughputs, inter-arrival times, lengths, windows.
LOG_PATTERN = re.compile(r"(BYTES|PKTS|DURATION|THROUGHPUT|IAT|LEN|WIN_MAX|PKT)", re.IGNORECASE)


def log_columns(feature_cols):
    return [c for c in feature_cols if LOG_PATTERN.search(c) and not c.startswith("proto_is_")]


SESSION_GAP_SECONDS = 3600


def session_split(window_ids, window_seconds, edge_window, labels, val_share, seed, val_unit="flows"):
    """Train on the first capture session and test on all of the second. In
    NF-UNSW-NB15-v3 every attack of session 1 falls in its first two hours, so
    a final block of session 1 would hold no attack flows to select models on.
    Validation is therefore a random share of session 1. With val_unit="flows"
    it is a class-stratified share of the flows, which stay in their window
    graphs as under the random flow split, so a validation flow can share its
    window with training flows. With val_unit="windows" it is a share of whole
    windows, drawn separately from the windows with and without attack flows,
    so that no validation flow shares a window with a training flow."""
    starts = np.asarray(window_ids, dtype="int64") * int(window_seconds)
    gaps = np.flatnonzero(np.diff(starts) > SESSION_GAP_SECONDS)
    if len(gaps) == 0:
        raise ValueError("no gap between capture sessions found")
    first_test = int(gaps[0]) + 1
    split = np.full(len(labels), 2, dtype="int8")
    in_s1 = edge_window < first_test
    split[in_s1] = 0
    rng = np.random.default_rng(seed)
    if val_unit == "windows":
        s1_windows = np.arange(first_test)
        has_attack = np.zeros(first_test, dtype=bool)
        has_attack[np.unique(edge_window[in_s1 & (labels > 0)])] = True
        val_windows = []
        for group in (s1_windows[has_attack], s1_windows[~has_attack]):
            n_val = int(round(len(group) * val_share))
            val_windows.append(rng.permutation(group)[:n_val])
        split[in_s1 & np.isin(edge_window, np.concatenate(val_windows))] = 1
        return split
    for cls in np.unique(labels[in_s1]):
        idx = np.flatnonzero(in_s1 & (labels == cls))
        n_val = int(round(len(idx) * val_share))
        split[rng.permutation(idx)[:n_val]] = 1
    return split


def aligned_split(window_ids, window_seconds, reference_dir):
    """Give windows of another length the same training, validation and test
    periods as the 60-s reference graphs: a window goes to the split in which
    its start time falls."""
    ref = np.load(reference_dir / "graphs.npz")
    ref_split = np.load(reference_dir / "edge_split.npy")
    with open(reference_dir / "meta.json", "r", encoding="utf-8") as handle:
        ref_seconds = int(json.load(handle)["window_seconds"])
    ref_code = ref_split[ref["edge_offsets"][:-1]]  # one code per reference window
    ref_start = ref["window_ids"].astype("int64") * ref_seconds
    val_start = ref_start[np.flatnonzero(ref_code == 1)[0]]
    test_start = ref_start[np.flatnonzero(ref_code == 2)[0]]
    start = np.asarray(window_ids, dtype="int64") * int(window_seconds)
    code = np.zeros(len(window_ids), dtype="int8")
    code[start >= val_start] = 1
    code[start >= test_start] = 2
    return code


def main():
    parser = argparse.ArgumentParser(description="Build time-windowed graphs and causal host context")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--split", choices=["chronological", "blocked", "random", "random_flow", "session", "aligned"],
                        default=None, help="random assigns whole windows at random; aligned copies the periods of --align-to")
    parser.add_argument("--align-to", default="graphs", help="reference graphs for --split aligned")
    parser.add_argument("--session-val", choices=["flows", "windows"], default="flows",
                        help="with --split session: hold out random session-1 flows or whole windows for validation")
    parser.add_argument("--no-structural", action="store_true")
    parser.add_argument("--no-behavioural", action="store_true")
    parser.add_argument("--no-role", action="store_true")
    parser.add_argument("--with-time", action="store_true")
    parser.add_argument("--age-mode", choices=["elapsed", "active"], default=None,
                        help="age as windows elapsed since a host was first seen (default) or as the number of "
                             "earlier windows in which it was active, which a gap in the capture leaves unchanged")
    parser.add_argument("--age-cap", type=float, default=None, help="cap the age count at this many windows")
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
    if args.age_mode is not None:
        overrides["context"]["age_mode"] = args.age_mode
    if args.age_cap is not None:
        overrides["context"]["age_cap"] = args.age_cap
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

    window_seconds = int(cfg.data.get("window_seconds", 60))
    if window_seconds != prepare_meta["window_seconds"]:
        df = assign_windows(df.drop(columns=["window_id"]), window_seconds)
        df = df.sort_values("start_ms", kind="mergesort").reset_index(drop=True)

    print("building windowed graphs ...", flush=True)
    t0 = time.perf_counter()
    graphs, row_index, context_names = build_windowed_graphs(df, cfg)
    build_seconds = time.perf_counter() - t0
    print("windows kept: {}".format(graphs.n_windows))
    print("edges: {}".format(len(graphs.edge_src)))
    print("context features: {} ({})".format(len(context_names), ", ".join(context_names)))
    print("graph + context build time: {:.1f}s ({:.2f} ms per window)".format(
        build_seconds, 1000 * build_seconds / max(graphs.n_windows, 1)))

    feature_cols = prepare_meta["feature_cols"]
    edge_features = df[feature_cols].values.astype("float32")[row_index]
    labels = df["target"].values.astype("int64")[row_index]

    logged = log_columns(feature_cols)
    for name in logged:
        idx = feature_cols.index(name)
        edge_features[:, idx] = np.log1p(np.clip(edge_features[:, idx], 0, None))

    mode = cfg.split.get("mode", "chronological")
    edge_window = graphs.edge_window
    if mode == "random_flow":
        edge_split = random_flow_split(labels, cfg, int(cfg.train.get("seed", 42)))
    elif mode == "random":
        rng = np.random.default_rng(int(cfg.train.get("seed", 42)))
        order = rng.permutation(graphs.n_windows)
        n_tr = int(graphs.n_windows * cfg.split["train"])
        n_va = int(graphs.n_windows * cfg.split["val"])
        window_code = np.empty(graphs.n_windows, dtype="int8")
        window_code[order[:n_tr]] = 0
        window_code[order[n_tr:n_tr + n_va]] = 1
        window_code[order[n_tr + n_va:]] = 2
        edge_split = window_code[edge_window]
    elif mode == "session":
        val_share = float(cfg.split["val"]) / (float(cfg.split["train"]) + float(cfg.split["val"]))
        edge_split = session_split(graphs.window_ids, window_seconds, edge_window, labels, val_share,
                                   int(cfg.train.get("seed", 42)), val_unit=args.session_val)
    elif mode == "aligned":
        window_code = aligned_split(graphs.window_ids, window_seconds, work_dir / args.align_to)
        edge_split = window_code[edge_window]
    else:
        assignment = chronological_split(graphs.window_ids, cfg)
        code = {"train": 0, "val": 1, "test": 2}
        window_code = np.array([code[assignment[w]] for w in graphs.window_ids], dtype="int8")
        edge_split = window_code[edge_window]

    train_edge_mask = edge_split == 0
    if train_edge_mask.sum() == 0:
        raise ValueError("Training split is empty. Check split fractions in the config.")

    # Node statistics are fitted on the nodes of windows that contain training
    # flows. For the chronological split those are exactly the training windows.
    train_windows = np.unique(edge_window[train_edge_mask])
    train_node_mask = np.zeros(len(graphs.context), dtype=bool)
    for w in train_windows:
        train_node_mask[graphs.node_offsets[w]:graphs.node_offsets[w + 1]] = True

    def fit_apply(matrix, mask):
        mean = matrix[mask].mean(axis=0, keepdims=True)
        std = matrix[mask].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return ((matrix - mean) / std).astype("float32"), mean.ravel(), std.ravel()

    edge_features, edge_mean, edge_std = fit_apply(edge_features, train_edge_mask)
    raw_context = graphs.context.copy()
    graphs.context, ctx_mean, ctx_std = fit_apply(graphs.context, train_node_mask)

    graphs.save(out_dir)
    np.save(out_dir / "edge_features.npy", edge_features)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "row_index.npy", row_index)
    np.save(out_dir / "edge_split.npy", edge_split)
    np.save(out_dir / "context_raw.npy", raw_context.astype("float32"))

    class_names = prepare_meta["classes"]
    present = {}
    for split_name, code in (("train", 0), ("val", 1), ("test", 2)):
        counts = np.bincount(labels[edge_split == code], minlength=len(class_names))
        present[split_name] = {class_names[i]: int(counts[i]) for i in range(len(class_names))}

    split_windows = {}
    for split_name, code in (("train", 0), ("val", 1), ("test", 2)):
        split_windows[split_name] = int(len(np.unique(edge_window[edge_split == code])))

    save_meta(out_dir, {
        "context_names": context_names,
        "feature_cols": feature_cols,
        "log_transformed": logged,
        "classes": class_names,
        "n_windows": int(graphs.n_windows),
        "n_edges": int(len(edge_features)),
        "n_nodes_total": int(len(graphs.context)),
        "edge_dim": int(edge_features.shape[1]),
        "context_dim": int(graphs.context.shape[1]),
        "split_mode": mode,
        "session_val": args.session_val if mode == "session" else None,
        "age_mode": cfg.context.get("age_mode", "elapsed"),
        "age_cap": cfg.context.get("age_cap"),
        "window_seconds": window_seconds,
        "split_windows": split_windows,
        "class_counts": present,
        "build_seconds": build_seconds,
        "ms_per_window_build": 1000 * build_seconds / max(graphs.n_windows, 1),
        "context_mean": ctx_mean.tolist(),
        "context_std": ctx_std.tolist(),
        "flows_dropped_small_windows": int(len(df) - len(row_index)),
    })

    print("\nsplit windows: {}".format(split_windows))
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
