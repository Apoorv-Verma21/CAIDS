"""Turns the experiment outputs into the paper's tables (JSON for the Word
builder), the numbers quoted in the text and the data figures, so that no
reported number is typed by hand.

    python scripts/analyze.py --root D:/CAIDS_work

Writes code/paper/generated/tables_generated.json, values.json and figures/.
The script stops with an error if a result file it needs is missing.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SEEDS = [1, 2, 3, 4, 5]
SHORT = {"Benign": "Ben.", "Analysis": "Ana.", "Backdoor": "Bkd.", "DoS": "DoS", "Exploits": "Exp.",
         "Fuzzers": "Fuz.", "Generic": "Gen.", "Reconnaissance": "Rec.", "Shellcode": "Shc.", "Worms": "Wor."}
ATTACKERS = ["175.45.176.0", "175.45.176.1", "175.45.176.2", "175.45.176.3"]
LOW_VOLUME = ["Analysis", "Backdoor", "Shellcode", "Worms"]

# result file, label, short key used in values.json
MAIN_ROWS = [
    ("baseline_random_forest", "Random forest (B1)", "rf"),
    ("baseline_xgboost", "XGBoost (B1)", "xgb"),
    ("gnn_flow_mlp", "Flow-only MLP (B2)", "mlp"),
    ("gnn_egraphsage", "E-GraphSAGE-style GNN (B3)", "b3"),
    ("gnn_centrality_egraphsage", "Centrality-initialised GNN (B4)", "b4"),
    ("gnn_cagids_concat", "CAG-IDS, concatenation", "concat"),
    ("gnn_cagids", "CAG-IDS, gated", "cagids"),
    ("gnn_cagids_gatv2", "CAG-IDS, gated, GATv2", "gatv2"),
]
NEURAL = [("flow_mlp", "mlp", "Flow-only MLP (B2)"), ("egraphsage", "b3", "E-GraphSAGE-style (B3)"),
          ("centrality_egraphsage", "b4", "Centrality-init. (B4)"), ("cagids_concat", "concat", "CAG-IDS, concat."),
          ("cagids", "cagids", "CAG-IDS, gated")]

# (label, experiment name in models.EXPERIMENT_MODELS, node initialisation, fusion, backbone, context dim)
VARIANTS = [
    ("Flow-only MLP (B2)", "flow_mlp", "none (no graph)", "none", "none", 17),
    ("E-GraphSAGE-style (B3)", "egraphsage", "constant vector", "none", "mean, *K* = 2", 17),
    ("Centrality-initialised (B4)", "centrality_egraphsage", "C1 only, encoded", "none", "mean, *K* = 2", 17),
    ("CAG-IDS, concatenation", "cagids_concat", "C1 + C2 + C3, encoded", "concatenation", "mean, *K* = 2", 17),
    ("CAG-IDS, gated", "cagids", "C1 + C2 + C3, encoded", "gate", "mean, *K* = 2", 17),
    ("CAG-IDS, GATv2", "cagids_gatv2", "C1 + C2 + C3, encoded", "gate", "GATv2, 4 heads, *K* = 2", 17),
    ("C1 + gate (control)", "cagids", "C1 only, encoded", "gate", "mean, *K* = 2", 5),
    ("Context-only probe", "context_only", "C1 + C2 + C3, encoded", "none", "none (*K* = 0)", 17),
]


class Missing(Exception):
    pass


def load(results, name, required=True):
    path = results / (name + ".json")
    if not path.exists():
        if required:
            raise Missing(str(path))
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def agg_of(results, name):
    return load(results, name)["aggregate"]


def fmt(agg, key, digits=3, pct=False):
    m, s = agg[key]["mean"], agg[key]["std"]
    if pct:
        return "{:.2f} ± {:.2f}".format(100 * m, 100 * s)
    return "{:.{d}f} ± {:.{d}f}".format(m, s, d=digits)


def mean_of(agg, key, digits=3):
    return "{:.{d}f}".format(agg[key]["mean"], d=digits)


def int_fmt(x):
    return "{:,}".format(int(x))


def pct_fmt(x, digits=1):
    return "{:.{d}f}%".format(100 * x, d=digits)


def num_word(n):
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return words[n] if 0 <= n < len(words) else int_fmt(n)


def signed(x, digits=3):
    return "{:+.{d}f}".format(x, d=digits).replace("-", "−")


def interval(lo, hi, digits=3):
    return "[{}, {}]".format(signed(lo, digits), signed(hi, digits))


# ----------------------------------------------------------------------------
# paired moving-block bootstrap over test windows

def load_pred(path):
    data = np.load(path)
    ids = data["edge_ids"]
    order = np.argsort(ids)
    return ids[order], data["y_true"].astype(int)[order], data["y_pred"].astype(int)[order]


def window_counts(ids, y, p, edge_window, windows, n_classes):
    """True positives, false positives and false negatives per class and window,
    on the rows of a fixed, shared window index."""
    w_idx = np.searchsorted(windows, edge_window[ids])
    tp = np.zeros((len(windows), n_classes))
    fp = np.zeros((len(windows), n_classes))
    fn = np.zeros((len(windows), n_classes))
    hit = y == p
    np.add.at(tp, (w_idx[hit], y[hit]), 1)
    np.add.at(fp, (w_idx[~hit], p[~hit]), 1)
    np.add.at(fn, (w_idx[~hit], y[~hit]), 1)
    return tp, fp, fn


def macro_from_counts(tp, fp, fn):
    support = tp + fn
    present = support > 0
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return f1[present].mean() if present.any() else np.nan


def paired_bootstrap(preds_a, preds_b, edge_window, n_classes, keep_ids=None, n_boot=2000, block=10, seed=0):
    """preds_a, preds_b: lists (one per seed) of (ids, y_true, y_pred).
    keep_ids restricts both models to the same flows. Returns the point estimate
    and 95% interval of the seed-averaged difference A - B, for macro-F1 over
    all classes and over the attack classes."""
    def restrict(pred):
        ids, y, p = pred
        if keep_ids is None:
            return ids, y, p
        m = np.isin(ids, keep_ids, assume_unique=True)
        return ids[m], y[m], p[m]

    a = [restrict(x) for x in preds_a]
    b = [restrict(x) for x in preds_b]
    for x in a[1:] + b:
        if not np.array_equal(x[0], a[0][0]):
            raise ValueError("the two models were scored on different flows")
    windows = np.unique(edge_window[a[0][0]])
    ca = [window_counts(i, y, p, edge_window, windows, n_classes) for i, y, p in a]
    cb = [window_counts(i, y, p, edge_window, windows, n_classes) for i, y, p in b]
    n = len(windows)

    def score(counts, idx, attacks):
        vals = []
        sl = slice(1, None) if attacks else slice(None)
        for tp, fp, fn in counts:
            vals.append(macro_from_counts(tp[idx].sum(0)[sl], fp[idx].sum(0)[sl], fn[idx].sum(0)[sl]))
        return float(np.mean(vals))

    full = np.arange(n)
    out = {}
    rng = np.random.default_rng(seed)
    starts_all = np.arange(max(n - block + 1, 1))
    n_blocks = int(np.ceil(n / block))
    samples = []
    for _ in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        samples.append(idx[idx < n])
    for attacks, key in ((False, "all"), (True, "attacks")):
        point = score(ca, full, attacks) - score(cb, full, attacks)
        deltas = np.array([score(ca, idx, attacks) - score(cb, idx, attacks) for idx in samples])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out[key] = {"a": score(ca, full, attacks), "b": score(cb, full, attacks), "delta": point,
                    "lo": float(lo), "hi": float(hi), "p_le_0": float((deltas <= 0).mean())}
    out["n_windows"] = n
    out["n_flows"] = int(len(a[0][0]))
    return out


def pred_files(pred_dir, tag):
    files = [pred_dir / "{}_seed{}.npz".format(tag, s) for s in SEEDS]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise Missing(", ".join(missing))
    return [load_pred(f) for f in files]


# ----------------------------------------------------------------------------
# descriptive facts quoted in the text

def dataset_facts(work, graph_meta, diag):
    values = {}
    df = pd.read_parquet(work / "flows.parquet",
                         columns=["src_ip", "dst_ip", "label", "attack", "start_ms", "probe__MIN_TTL"])
    n_raw = len(df)
    benign = df[df["label"] == 0]
    attack = df[df["label"] == 1]
    values["raw_flows"] = int_fmt(n_raw)
    values["raw_benign"] = int_fmt(len(benign))
    values["raw_attack"] = int_fmt(len(attack))
    values["attack_share"] = "{:.2f}%".format(100 * len(attack) / n_raw)

    att_src = set(attack["src_ip"].unique())
    victims = set(attack["dst_ip"].unique())
    values["n_attack_sources"] = int_fmt(len(att_src))
    values["n_victims"] = int_fmt(len(victims))
    values["benign_to_victims"] = int_fmt(benign["dst_ip"].isin(victims).sum())
    values["benign_to_attackers"] = int_fmt(benign["dst_ip"].isin(att_src).sum())
    values["benign_from_attackers"] = int_fmt(benign["src_ip"].isin(att_src).sum())
    values["benign_victim_to_attacker"] = int_fmt((benign["src_ip"].isin(victims) & benign["dst_ip"].isin(att_src)).sum())
    clients = ["59.166.0.{}".format(i) for i in range(10)]
    servers = ["149.171.126.{}".format(i) for i in range(10)]
    core = ((benign["src_ip"].isin(clients) & benign["dst_ip"].isin(servers)) |
            (benign["src_ip"].isin(servers) & benign["dst_ip"].isin(clients)))
    values["benign_core_share"] = pct_fmt(core.mean())

    ct = pd.crosstab(attack["src_ip"], attack["attack"])
    for cls, key in [("Analysis", "analysis"), ("Worms", "worms")]:
        values["attacker_{}_min".format(key)] = int_fmt(ct[cls].min())
        values["attacker_{}_max".format(key)] = int_fmt(ct[cls].max())

    ttl_b = benign["probe__MIN_TTL"].value_counts(normalize=True)
    ttl_a = attack["probe__MIN_TTL"].value_counts(normalize=True)
    values["ttl_benign_31"] = pct_fmt(ttl_b.get(31.0, 0.0))
    values["ttl_attack_254"] = pct_fmt(ttl_a.get(254.0, 0.0))
    values["ttl_attack_62"] = pct_fmt(ttl_a.get(62.0, 0.0))

    counts = attack["attack"].value_counts()
    for cls in counts.index:
        values["share_{}".format(cls.lower())] = pct_fmt(counts[cls] / len(attack))

    sessions = diag["dataset"]["sessions"]
    for i, s in enumerate(sessions[:2], start=1):
        values["s{}_hours".format(i)] = "{:.1f}".format(s["hours"])
        values["s{}_flows".format(i)] = int_fmt(s["flows"])
        values["s{}_attack_rate".format(i)] = pct_fmt(s["attack_flows"] / s["flows"])
    gap = pd.Timestamp(sessions[1]["start_utc"]) - pd.Timestamp(sessions[0]["end_utc"])
    values["session_gap_days"] = "{:.0f}".format(gap.total_seconds() / 86400)
    sp = diag["splits"]
    t = {k: (pd.Timestamp(sp[k]["start_utc"]), pd.Timestamp(sp[k]["end_utc"])) for k in sp}
    values["train_end"] = t["train"][1].strftime("%H:%M")
    values["test_start"] = t["test"][0].strftime("%H:%M")
    values["test_end"] = t["test"][1].strftime("%H:%M")
    s2_start = pd.Timestamp(sessions[1]["start_utc"])
    values["train_s2_hours"] = "{:.1f}".format((t["train"][1] - s2_start).total_seconds() / 3600)
    values["val_hours"] = "{:.1f}".format((t["val"][1] - t["val"][0]).total_seconds() / 3600)
    values["test_hours"] = "{:.1f}".format((t["test"][1] - t["test"][0]).total_seconds() / 3600)
    values["train_attack_flows"] = int_fmt(sp["train"]["attack_flows"])
    values["test_attack_flows"] = int_fmt(sp["test"]["attack_flows"])

    row_index = np.load(work / "graphs" / "row_index.npy")
    edge_split = np.load(work / "graphs" / "edge_split.npy")
    minute = ((df["start_ms"].to_numpy()[row_index] // 60000) % 1440).astype(int)
    train_minutes = set(np.unique(minute[edge_split == 0]))
    values["test_clock_overlap"] = pct_fmt(np.isin(minute[edge_split == 2], list(train_minutes)).mean())

    ctx = np.load(work / "graphs" / "context_raw.npy")
    cn = graph_meta["context_names"]
    names = host_names(work, df, row_index)
    is_att = np.isin(names, ATTACKERS)
    values["host_window_pairs"] = int_fmt(len(ctx))
    nov = ctx[:, cn.index("peer_novelty")]
    first = ctx[:, cn.index("is_first_seen")] > 0.5
    values["novelty_zero_share"] = pct_fmt((nov == 0).mean())
    values["first_seen_pairs"] = int_fmt(first.sum())
    values["n_private_hosts"] = int_fmt(len(set(names[ctx[:, cn.index("is_private")] > 0.5])))
    od = ctx[:, cn.index("out_degree")]
    ent = ctx[:, cn.index("dst_port_entropy")]
    olo, ohi = np.percentile(od[is_att], [5, 95])
    elo, ehi = np.percentile(ent[is_att], [5, 95])
    band = (od >= olo) & (od <= ohi) & (ent >= elo) & (ent <= ehi)
    values["band_attackers"] = pct_fmt(band[is_att].mean())
    values["band_others"] = pct_fmt(band[~is_att].mean())
    values["band_others_n"] = int_fmt(band[~is_att].sum())
    values["max_log_age"] = "{:.1f}".format(ctx[:, cn.index("log_age_windows")].max())

    values["n_edge_features"] = int_fmt(graph_meta["edge_dim"])
    values["n_retained_numeric"] = int_fmt(len([c for c in graph_meta["feature_cols"] if not c.startswith("proto_is_")]))
    values["n_log_features"] = int_fmt(len(graph_meta["log_transformed"]))
    values["n_context_features"] = int_fmt(graph_meta["context_dim"])
    return values


_HOST_NAMES = {}


def host_names(work, df, row_index):
    """IP address of every host-window node, in the node order of the graphs."""
    key = str(work)
    if key in _HOST_NAMES:
        return _HOST_NAMES[key]
    g = np.load(work / "graphs" / "graphs.npz")
    flows = df.iloc[row_index]
    src = flows["src_ip"].to_numpy(dtype=object)
    dst = flows["dst_ip"].to_numpy(dtype=object)
    eo, no = g["edge_offsets"], g["node_offsets"]
    names = np.empty(no[-1], dtype=object)
    for w in range(len(eo) - 1):
        names[no[w]:no[w + 1]] = np.unique(np.concatenate([src[eo[w]:eo[w + 1]], dst[eo[w]:eo[w + 1]]]))
    _HOST_NAMES[key] = names
    return names


def parameter_counts(meta):
    from cagids.config import load_config
    from cagids.models import build_model
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    out = {}
    for label, name, _, _, _, ctx_dim in VARIANTS:
        model_cfg = dict(cfg["model"])
        model_cfg.update({"name": name, "structural_dim": 5})
        run_cfg = type(cfg)({**cfg, "model": model_cfg})
        model = build_model(meta["edge_dim"], ctx_dim, len(meta["classes"]), run_cfg)
        out[label] = int(sum(p.numel() for p in model.parameters()))
    return out


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=os.path.join(ROOT, "paper", "generated"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    work = root / "outputs" / "work"
    results = root / "outputs" / "results"
    pred_dir = results / "predictions"
    out_dir = Path(args.out)
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(work / "graphs" / "meta.json", encoding="utf-8"))
    prepare_meta = json.load(open(work / "prepare_meta.json", encoding="utf-8"))
    classes = meta["classes"]
    n_classes = len(classes)
    diag = load(results, "diagnostics_graphs")
    tables, values = {}, {}
    edge_window = np.load(work / "graphs" / "graphs.npz")["edge_window"]

    # ---------------- dataset composition ----------------
    cc = meta["class_counts"]
    rows = []
    for c in classes:
        total = cc["train"][c] + cc["val"][c] + cc["test"][c]
        rows.append([c, int_fmt(cc["train"][c]), int_fmt(cc["val"][c]), int_fmt(cc["test"][c]), int_fmt(total)])
    tot = {s: sum(cc[s].values()) for s in ("train", "val", "test")}
    rows.append(["Total flows", int_fmt(tot["train"]), int_fmt(tot["val"]), int_fmt(tot["test"]), int_fmt(sum(tot.values()))])
    sw = meta["split_windows"]
    rows.append(["Windows (60 s)", int_fmt(sw["train"]), int_fmt(sw["val"]), int_fmt(sw["test"]), int_fmt(meta["n_windows"])])
    sp = diag["splits"]

    def period(s):
        a, b = pd.Timestamp(sp[s]["start_utc"]), pd.Timestamp(sp[s]["end_utc"])
        if a.date() == b.date():
            return "{} {} to {}".format(a.strftime("%d %b"), a.strftime("%H:%M"), b.strftime("%H:%M"))
        return "{} {} to {} {}".format(a.strftime("%d %b"), a.strftime("%H:%M"), b.strftime("%d %b"), b.strftime("%H:%M"))
    rows.append(["Period (UTC, 2015)", period("train"), period("val"), period("test"), ""])
    tables["dataset"] = {
        "caption": "Class composition of NF-UNSW-NB15-v3 under the chronological split",
        "colWidths": [2.2, 2.6, 1.7, 1.7, 1.4],
        "header": [["Class", "Train", "Validation", "Test", "Total"]],
        "rows": rows,
        "midRules": [len(classes) - 1],
        "fontSize": 17,
        "align": ["left", "right", "right", "right", "right"],
        "notes": "Windows with fewer than two flows are dropped ({} flows in total). Training, validation and test are the first 70%, the next 10% and the last 20% of the non-empty 60-s windows in time order.".format(
            ["no", "one", "two", "three"][meta["flows_dropped_small_windows"]] if meta["flows_dropped_small_windows"] < 4
            else int_fmt(meta["flows_dropped_small_windows"])),
    }
    values.update({
        "n_flows": int_fmt(sum(tot.values())), "n_windows": int_fmt(meta["n_windows"]),
        "n_train_flows": int_fmt(tot["train"]), "n_val_flows": int_fmt(tot["val"]), "n_test_flows": int_fmt(tot["test"]),
        "n_train_windows": int_fmt(sw["train"]), "n_val_windows": int_fmt(sw["val"]), "n_test_windows": int_fmt(sw["test"]),
        "build_ms": "{:.1f}".format(meta["ms_per_window_build"]), "flows_dropped": int_fmt(meta["flows_dropped_small_windows"]),
        "share_benign_test": pct_fmt(cc["test"]["Benign"] / tot["test"]),
        "n_test_attack": int_fmt(tot["test"] - cc["test"]["Benign"]),
    })
    for c in classes:
        values["test_count_{}".format(c.lower())] = int_fmt(cc["test"][c])
        values["train_count_{}".format(c.lower())] = int_fmt(cc["train"][c])
    train_att = tot["train"] - cc["train"]["Benign"]
    test_att = tot["test"] - cc["test"]["Benign"]
    values["train_share_backdoor"] = pct_fmt(cc["train"]["Backdoor"] / train_att)
    values["test_share_backdoor"] = pct_fmt(cc["test"]["Backdoor"] / test_att)
    values["n_train_attack"] = int_fmt(train_att)

    # ---------------- flow-feature handling ----------------
    ident = [prepare_meta["resolved"][r] for r in ("src_ip", "dst_ip", "src_port", "dst_port", "start_time", "end_time")
             if r in prepare_meta["resolved"]]
    excluded = [c.replace("probe__", "") for c in prepare_meta["probe_cols"]]
    kept = [c for c in meta["feature_cols"] if not c.startswith("proto_is_")]
    logged = meta["log_transformed"]
    tables["features"] = {
        "caption": "Use of the NetFlow fields of NF-UNSW-NB15-v3",
        "colWidths": [2.6, 0.8, 5.6],
        "header": [["Group", "Fields", "Use"]],
        "rows": [
            ["Endpoints and time", str(len(ident)), "Build the window graphs and the host context; never part of the flow vector (addresses, ports, start and end timestamps)"],
            ["Labels", "2", "Targets only (Label, Attack)"],
            ["Excluded shortcut fields", str(len(excluded)), "Removed from all models, kept for the audit: MIN_TTL, MAX_TTL and the two per-second byte rates"],
            ["Retained numeric fields", str(len(kept)), "Flow vector; {} heavy-tailed fields transformed with log(1 + *x*)".format(len(logged))],
            ["Protocol indicators", "3", "Flow vector; TCP, UDP and ICMP indicators derived from PROTOCOL"],
            ["Flow vector **e**_{*f*}", str(meta["edge_dim"]), "Standardised with training-period means and deviations"],
        ],
        "midRules": [4],
        "fontSize": 17,
        "align": ["left", "center", "left"],
        "notes": "Categorical codes (L7_PROTO, ICMP_TYPE, DNS_QUERY_TYPE, FTP_COMMAND_RET_CODE) and DNS_QUERY_ID are retained as numeric values, as in the released feature set.",
    }
    values["inf_rate_flows"] = int_fmt(np.isinf(pd.read_parquet(
        work / "flows.parquet", columns=["probe__SRC_TO_DST_SECOND_BYTES", "probe__DST_TO_SRC_SECOND_BYTES"]).to_numpy()).any(axis=1).sum())

    # ---------------- variants with parameter counts ----------------
    params = parameter_counts(meta)
    rows = []
    for label, name, init, fusion, backbone, _ in VARIANTS:
        rows.append([{"text": label, "bold": False}, init, fusion, backbone, int_fmt(params[label])])
    tables["variants"] = {
        "caption": "Model variants. All share one implementation, the training loop and the loss",
        "colWidths": [2.9, 2.5, 1.5, 2.1, 1.3],
        "header": [["Variant", "Node initialisation", "Fusion", "Backbone", "Parameters"]],
        "rows": rows,
        "midRules": [5],
        "fontSize": 17,
        "align": ["left", "left", "center", "center", "right"],
        "notes": "All graph variants pass the standardised flow vector to the classifier as an additional input, except the context-only probe, which sees only the two encoded endpoint contexts. The C1 + gate control is gated CAG-IDS trained on graphs whose context contains only C1. Parameter counts are computed from the model definitions for the 46 flow features and the 10 classes.",
    }
    for label, name, *_ in VARIANTS:
        values["params_" + ("c1_gate" if label.startswith("C1") else name)] = int_fmt(params[label])

    # ---------------- diagnostics ----------------
    ds = diag["dataset"]
    values["n_hosts"] = int_fmt(ds["unique_hosts"])
    values["hosts_per_window_median"] = "{:.0f}".format(ds["hosts_per_window"]["median"])
    values["hosts_per_window_max"] = int_fmt(ds["hosts_per_window"]["max"])
    values["flows_per_window_median"] = int_fmt(ds["flows_per_window"]["median"])
    values["flows_per_window_max"] = int_fmt(ds["flows_per_window"]["max"])
    ho = diag["host_overlap"]
    values["test_hosts_seen"] = pct_fmt(ho["test_hosts_seen_in_train"])
    values["test_attack_from_train_sources"] = pct_fmt(ho["test_attack_flows_from_train_attack_sources"])
    values["hosts_test"] = int_fmt(ho["hosts_test"])

    ip, sf, cp = diag["identity_probes"], diag["single_feature_probes"], diag["context_probes"]
    rows = []
    label = {"src_ip": "Source address only", "dst_ip": "Destination address only", "dst_port": "Destination port only",
             "src_ip+dst_ip": "Address pair", "src_ip+dst_ip+dst_port": "Address pair and destination port"}
    for k in ["src_ip", "dst_ip", "dst_port", "src_ip+dst_ip", "src_ip+dst_ip+dst_port"]:
        s = ip[k]
        rows.append([label[k], "{:.3f}".format(s["macro_f1"]), "{:.3f}".format(s["macro_f1_attacks"]),
                     "{:.3f}".format(s["binary_f1"]), "{:.4f}".format(s["false_positive_rate"])])
    n_identity = len(rows)
    for k, name in [("probe__MIN_TTL", "MIN_TTL (excluded)"), ("probe__MAX_TTL", "MAX_TTL (excluded)"),
                    ("probe__SRC_TO_DST_SECOND_BYTES", "SRC_TO_DST_SECOND_BYTES (excluded)"),
                    ("probe__DST_TO_SRC_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES (excluded)")]:
        s = sf[k]
        rows.append([name, "{:.3f}".format(s["macro_f1"]), "{:.3f}".format(s["macro_f1_attacks"]),
                     "{:.3f}".format(s["binary_f1"]), "{:.4f}".format(s["false_positive_rate"])])
    kept_probes = sorted([(k, v) for k, v in sf.items() if not k.startswith("probe__")], key=lambda kv: -kv[1]["macro_f1"])
    for k, s in kept_probes[:3]:
        rows.append(["{} (retained)".format(k), "{:.3f}".format(s["macro_f1"]), "{:.3f}".format(s["macro_f1_attacks"]),
                     "{:.3f}".format(s["binary_f1"]), "{:.4f}".format(s["false_positive_rate"])])
    n_kept = len(rows)
    for k, name in [("context_only_rf", "RF on endpoint context only"), ("flow_only_rf", "RF on flow features only"),
                    ("flow_plus_context_rf", "RF on flow features and endpoint context")]:
        s = cp[k]
        rows.append([name, "{:.3f} ± {:.3f}".format(s["macro_f1"]["mean"], s["macro_f1"]["std"]),
                     "{:.3f}".format(s["macro_f1_attacks"]["mean"]), "{:.3f}".format(s["binary_f1"]["mean"]),
                     "{:.4f}".format(s["false_positive_rate"]["mean"])])
    tables["diagnostics"] = {
        "caption": "Shortcut and information probes on the chronological test split",
        "colWidths": [4.4, 1.6, 1.4, 1.1, 1.1],
        "header": [["Probe", "Macro-F1", "Attack macro-F1", "Binary F1", "FPR"]],
        "rows": rows,
        "midRules": [n_identity - 1, n_kept - 1],
        "fontSize": 17,
        "align": ["left", "center", "center", "center", "center"],
        "notes": "Identity probes predict the majority training class of each key (unseen keys: Benign). Single-feature probes are depth-6 decision trees trained on all training attack flows and 300,000 benign training flows. Context probes are 200-tree random forests trained on the same 400,000-flow subsample as B1 (mean ± SD over three seeds). Attack macro-F1 averages the nine attack classes; binary F1 treats all attack classes as one.",
    }
    values["probe_src_binary"] = "{:.3f}".format(ip["src_ip"]["binary_f1"])
    values["probe_dst_binary"] = "{:.3f}".format(ip["dst_ip"]["binary_f1"])
    values["probe_src_macro"] = "{:.3f}".format(ip["src_ip"]["macro_f1"])
    values["probe_src_fpr"] = "{:.4f}".format(ip["src_ip"]["false_positive_rate"])
    values["probe_ttl_binary"] = "{:.3f}".format(sf["probe__MIN_TTL"]["binary_f1"])
    values["probe_maxttl_binary"] = "{:.3f}".format(sf["probe__MAX_TTL"]["binary_f1"])
    values["probe_rate_s2d_binary"] = "{:.3f}".format(sf["probe__SRC_TO_DST_SECOND_BYTES"]["binary_f1"])
    values["probe_rate_d2s_binary"] = "{:.3f}".format(sf["probe__DST_TO_SRC_SECOND_BYTES"]["binary_f1"])
    values["probe_shortest_binary"] = "{:.3f}".format(sf["SHORTEST_FLOW_PKT"]["binary_f1"])
    values["probe_minlen_binary"] = "{:.3f}".format(sf["MIN_IP_PKT_LEN"]["binary_f1"])
    values["probe_best_kept_name"] = kept_probes[0][0]
    values["probe_best_kept_macro"] = "{:.3f}".format(kept_probes[0][1]["macro_f1"])
    for k in ("context_only_rf", "flow_only_rf", "flow_plus_context_rf"):
        values[k + "_macro"] = "{:.3f}".format(cp[k]["macro_f1"]["mean"])
        values[k + "_attack"] = "{:.3f}".format(cp[k]["macro_f1_attacks"]["mean"])
        values[k + "_binary"] = "{:.3f}".format(cp[k]["binary_f1"]["mean"])
    values["context_drop_points"] = "{:.1f}".format(
        100 * (cp["flow_only_rf"]["macro_f1"]["mean"] - cp["flow_plus_context_rf"]["macro_f1"]["mean"]))
    values.update(dataset_facts(work, meta, diag))

    # ---------------- main results ----------------
    payloads = {name: load(results, name) for name, _, _ in MAIN_ROWS}
    best_macro = max(p["aggregate"]["macro_f1"]["mean"] for p in payloads.values())
    best_att = max(p["aggregate"]["macro_f1_attacks"]["mean"] for p in payloads.values())
    rows = []
    for name, label, key in MAIN_ROWS:
        agg = payloads[name]["aggregate"]
        rows.append([{"text": label, "bold": key == "cagids"},
                     {"text": fmt(agg, "macro_f1"), "bold": agg["macro_f1"]["mean"] == best_macro},
                     {"text": fmt(agg, "macro_f1_attacks"), "bold": agg["macro_f1_attacks"]["mean"] == best_att},
                     fmt(agg, "weighted_f1"), fmt(agg, "binary_f1"), fmt(agg, "false_positive_rate", pct=True), fmt(agg, "pr_auc")])
        for metric, short in [("macro_f1", "macro"), ("macro_f1_attacks", "attack"), ("weighted_f1", "weighted"),
                              ("binary_f1", "binary"), ("pr_auc", "ap")]:
            values["m_{}_{}".format(key, short)] = fmt(agg, metric)
            values["m_{}_{}_mean".format(key, short)] = mean_of(agg, metric)
        values["m_{}_macro_sd".format(key)] = "{:.3f}".format(agg["macro_f1"]["std"])
        values["m_{}_fpr".format(key)] = fmt(agg, "false_positive_rate", pct=True)
        values["m_{}_fpr_mean".format(key)] = "{:.2f}%".format(100 * agg["false_positive_rate"]["mean"])
        for c in classes:
            values["pc_{}_{}".format(key, c.lower())] = "{:.3f}".format(agg["per_class"][c]["f1_mean"])
        runs = payloads[name]["runs"]
        for c in classes:
            values["prec_{}_{}".format(key, c.lower())] = "{:.3f}".format(np.mean([r["per_class"][c]["precision"] for r in runs]))
            values["rec_{}_{}".format(key, c.lower())] = "{:.3f}".format(np.mean([r["per_class"][c]["recall"] for r in runs]))
        if "best_epoch" in runs[0]:
            values["epochs_{}".format(key)] = ", ".join(str(r["best_epoch"]) for r in runs)
            values["epochs_{}_median".format(key)] = "{:.0f}".format(np.median([r["best_epoch"] for r in runs]))
    tables["main"] = {
        "caption": "Detection performance on the chronological test split (mean ± SD over five seeds)",
        "colWidths": [2.75, 1.45, 1.45, 1.45, 1.45, 1.2, 1.45],
        "header": [["Model", "Macro-F1", "Attack macro-F1", "Weighted F1", "Binary F1", "FPR (%)", "AP"]],
        "rows": rows,
        "midRules": [1, 2],
        "fontSize": 16,
        "align": ["left", "center", "center", "center", "center", "center", "center"],
        "notes": "Macro-F1 averages the ten classes present in the test period; attack macro-F1 averages the nine attack classes. FPR: share of benign test flows assigned to any attack class. AP: average precision of the attack score defined in [sec:metrics]. Bold: best mean in the column.",
    }
    aggs_main = [p["aggregate"] for p in payloads.values()]
    values["max_f1_hard_classes"] = "{:.3f}".format(max(a["per_class"][c]["f1_mean"] for a in aggs_main
                                                        for c in ("Backdoor", "Analysis", "DoS")))
    values["main_min_binary"] = "{:.3f}".format(min(a["binary_f1"]["mean"] for a in aggs_main))
    values["main_min_ap"] = "{:.3f}".format(min(a["pr_auc"]["mean"] for a in aggs_main))
    values["main_max_fpr"] = "{:.2f}%".format(100 * max(a["false_positive_rate"]["mean"] for a in aggs_main))
    values["main_min_weighted"] = "{:.3f}".format(min(a["weighted_f1"]["mean"] for a in aggs_main))
    neural_best = max(((k, payloads[n]["aggregate"]["macro_f1"]["mean"]) for n, _, k in MAIN_ROWS if n.startswith("gnn_")),
                      key=lambda kv: kv[1])
    values["neural_best_key"] = neural_best[0]
    values["graph_best_macro"] = "{:.3f}".format(max(payloads[n]["aggregate"]["macro_f1"]["mean"]
                                                     for n, _, k in MAIN_ROWS if n.startswith("gnn_") and k != "mlp"))
    all_epochs = []
    for tag, *_ in NEURAL + [("cagids_gatv2", "gatv2", "")]:
        all_epochs += [r["best_epoch"] for r in load(results, "gnn_" + tag)["runs"]]
    values["best_epoch_median_all"] = "{:.0f}".format(np.median(all_epochs))
    values["best_epoch_ge25"] = num_word(sum(e >= 25 for e in all_epochs))
    values["best_epoch_runs"] = int_fmt(len(all_epochs))

    # ---------------- rare-class precision and recall ----------------
    rows = []
    for c in ["Analysis", "Backdoor", "DoS", "Shellcode", "Worms"]:
        row = [c]
        for key in ("rf", "mlp", "cagids"):
            row.append("{} / {}".format(values["prec_{}_{}".format(key, c.lower())], values["rec_{}_{}".format(key, c.lower())]))
        row.append(values["test_count_{}".format(c.lower())])
        rows.append(row)
    tables["rareclass"] = {
        "caption": "Precision / recall of the five weakest attack classes on the chronological test split (mean over five seeds)",
        "colWidths": [2.2, 2.0, 2.0, 2.0, 1.4],
        "header": [["Class", "Random forest", "Flow-only MLP (B2)", "CAG-IDS, gated", "Test flows"]],
        "rows": rows,
        "fontSize": 17,
        "align": ["left", "center", "center", "center", "right"],
        "notes": "The neural models are trained with the inverse-frequency class weights of Eq. [eq:class_weight], which weight a Worms flow about {{weight_ratio_worms_benign}} times as heavily as a benign flow.",
    }
    tr_counts = np.array([cc["train"][c] for c in classes], dtype=float)
    w = tr_counts.sum() / (n_classes * tr_counts)
    values["weight_ratio_worms_benign"] = int_fmt(round(w[classes.index("Worms")] / w[0], -3))

    # ---------------- per-class F1 ----------------
    show = [("baseline_random_forest", "RF"), ("baseline_xgboost", "XGB"), ("gnn_flow_mlp", "B2"),
            ("gnn_egraphsage", "B3"), ("gnn_centrality_egraphsage", "B4"), ("gnn_cagids_concat", "Concat."),
            ("gnn_cagids", "Gated"), ("gnn_cagids_gatv2", "GATv2")]
    per_rows = []
    for c in classes:
        f1s = [payloads[name]["aggregate"]["per_class"][c]["f1_mean"] for name, _ in show]
        best = max(f1s)
        row = [c] + [{"text": "{:.3f}".format(v), "bold": v == best} for v in f1s]
        row.append(int_fmt(meta["class_counts"]["test"][c]))
        per_rows.append(row)
    tables["perclass"] = {
        "caption": "Per-class F1 on the chronological test split (mean over five seeds)",
        "colWidths": [2.1] + [0.95] * len(show) + [1.15],
        "header": [["Class"] + [s for _, s in show] + ["Test flows"]],
        "rows": per_rows,
        "fontSize": 16,
        "align": ["left"] + ["center"] * len(show) + ["right"],
        "notes": "RF random forest, XGB XGBoost (both B1), B2 flow-only MLP, B3 E-GraphSAGE-style GNN, B4 centrality-initialised GNN, Concat. and Gated the two CAG-IDS fusions, GATv2 gated CAG-IDS with the attention backbone. Bold: best in the row.",
    }

    # ---------------- confusion of the random forest ----------------
    rf = payloads["baseline_random_forest"]
    mats = []
    for r in rf["runs"]:
        m = np.array(r["confusion"], dtype="float64")
        mats.append(100 * m / np.maximum(m.sum(axis=1, keepdims=True), 1))
    conf = np.mean(mats, axis=0)
    tables["confusion"] = {
        "caption": "Row-normalised confusion matrix of the random forest on the chronological test split (%, mean over five seeds)",
        "colWidths": [2.4] + [0.95] * n_classes,
        "header": [["True class"] + [SHORT[c] for c in classes]],
        "rows": [[c] + [{"text": "{:.1f}".format(v) if v >= 0.05 else "0", "bold": i == j} for j, v in enumerate(conf[i])]
                 for i, c in enumerate(classes)],
        "fontSize": 16,
        "align": ["left"] + ["center"] * n_classes,
        "notes": "Each row shows how the test flows of one true class are distributed over the predicted classes; diagonal entries (bold) are recalls.",
    }
    idx = {c: i for i, c in enumerate(classes)}
    for a, b in [("Analysis", "Exploits"), ("Backdoor", "Backdoor"), ("Backdoor", "Exploits"), ("Backdoor", "Fuzzers"),
                 ("Backdoor", "Generic"), ("Backdoor", "Reconnaissance"), ("Fuzzers", "Backdoor"), ("DoS", "DoS"),
                 ("DoS", "Exploits"), ("DoS", "Fuzzers")]:
        values["rf_{}_to_{}".format(a.lower(), b.lower())] = "{:.1f}%".format(conf[idx[a], idx[b]])
    values["rf_backdoor_precision"] = "{:.3f}".format(np.mean([r["per_class"]["Backdoor"]["precision"] for r in rf["runs"]]))

    # ---------------- paired bootstrap, main comparisons ----------------
    preds = {}

    def P(tag):
        if tag not in preds:
            preds[tag] = pred_files(pred_dir, tag)
        return preds[tag]

    comparisons = [
        ("cagids", "egraphsage", "Gated CAG-IDS − E-GraphSAGE-style (H1)", "cagids_b3"),
        ("cagids", "centrality_egraphsage", "Gated CAG-IDS − centrality-initialised (H1)", "cagids_b4"),
        ("cagids", "cagids_concat", "Gate − concatenation (H3)", "cagids_concat"),
        ("cagids", "c1_gate", "Gated CAG-IDS − C1 + gate control", "cagids_c1gate"),
        ("c1_gate", "centrality_egraphsage", "C1 + gate control − centrality-initialised", "c1gate_b4"),
        ("cagids_gatv2", "cagids", "GATv2 − mean aggregation (both gated)", "gatv2_cagids"),
        ("egraphsage", "flow_mlp", "E-GraphSAGE-style − flow-only MLP", "b3_mlp"),
        ("centrality_egraphsage", "flow_mlp", "Centrality-initialised − flow-only MLP", "b4_mlp"),
        ("cagids", "flow_mlp", "Gated CAG-IDS − flow-only MLP", "cagids_mlp"),
        ("cagids", "random_forest", "Gated CAG-IDS − random forest", "cagids_rf"),
    ]
    # the same comparisons under the random flow split, on its own test flows
    comparisons += [
        ("rf_cagids", "rf_flow_mlp", "Gated CAG-IDS − flow-only MLP, random flow split", "rnd_cagids_mlp"),
        ("rf_cagids", "rf_centrality_egraphsage", "Gated CAG-IDS − centrality-initialised, random flow split", "rnd_cagids_b4"),
    ]
    edge_window_random = np.load(work / "graphs_random_flow" / "graphs.npz")["edge_window"]
    rows = []
    boot = {}
    for a, b, label, key in comparisons:
        windows_of = edge_window_random if a.startswith("rf_") else edge_window
        res = paired_bootstrap(P(a), P(b), windows_of, n_classes, n_boot=args.n_boot)
        boot[key] = res
        ra, rt = res["all"], res["attacks"]
        rows.append([label, signed(ra["delta"]), interval(ra["lo"], ra["hi"]), signed(rt["delta"]), interval(rt["lo"], rt["hi"])])
        values["boot_{}".format(key)] = "{} {}".format(signed(ra["delta"]), interval(ra["lo"], ra["hi"]))
        values["boot_{}_att".format(key)] = "{} {}".format(signed(rt["delta"]), interval(rt["lo"], rt["hi"]))
        values["boot_{}_delta".format(key)] = signed(ra["delta"])
        values["boot_{}_delta_abs".format(key)] = "{:.3f}".format(abs(ra["delta"]))
        values["boot_{}_ci".format(key)] = interval(ra["lo"], ra["hi"])
        values["boot_{}_att_delta".format(key)] = signed(rt["delta"])
        values["boot_{}_att_ci".format(key)] = interval(rt["lo"], rt["hi"])
    tables["significance"] = {
        "caption": "Paired differences in macro-F1, with 95% moving-block bootstrap intervals",
        "colWidths": [4.4, 1.2, 1.9, 1.3, 1.9],
        "header": [["Comparison (A − B)", "Δ macro-F1", "95% interval", "Δ attack macro-F1", "95% interval"]],
        "rows": rows,
        "midRules": [2, 5, 9],
        "fontSize": 17,
        "align": ["left", "center", "center", "center", "center"],
        "notes": "Δ is the seed-averaged score of model A minus that of model B, on the chronological test split except in the last two rows, which use the test flows of the random flow split. Blocks of ten consecutive test windows (10 min) are resampled 2,000 times. An interval that lies entirely above zero supports A.",
    }
    values["h1_b3_supported"] = "yes" if boot["cagids_b3"]["all"]["lo"] > 0 else "no"
    values["h1_b4_supported"] = "yes" if boot["cagids_b4"]["all"]["lo"] > 0 else "no"
    values["h3_supported"] = "yes" if boot["cagids_concat"]["all"]["lo"] > 0 else "no"

    # ---------------- context-group ablation (H2) ----------------
    abl = [("gnn_cagids", "Full context (C1 + C2 + C3)", "full"), ("gnn_abl_no_structural", "without C1 (structural)", "no_c1"),
           ("gnn_abl_no_behavioural", "without C2 (behavioural)", "no_c2"), ("gnn_abl_no_role", "without C3 (role)", "no_c3"),
           ("gnn_c1_gate", "C1 only (C1 + gate control)", "c1_only"), ("gnn_abl_with_time", "with C4 (time of day)", "with_c4")]
    rows = []
    low_means = {}
    for name, label, key in abl:
        p = load(results, name)
        agg = p["aggregate"]
        per_seed_low = [np.mean([r["per_class"][c]["f1"] for c in LOW_VOLUME]) for r in p["runs"]]
        low_means[key] = (float(np.mean(per_seed_low)), float(np.std(per_seed_low, ddof=1)))
        row = [label, fmt(agg, "macro_f1"), fmt(agg, "macro_f1_attacks")]
        for c in LOW_VOLUME:
            row.append("{:.3f}".format(agg["per_class"][c]["f1_mean"]))
        row.append("{:.3f} ± {:.3f}".format(*low_means[key]))
        row.append("{:.3f}".format(agg["per_class"]["Reconnaissance"]["f1_mean"]))
        rows.append(row)
        values["abl_{}_macro".format(key)] = fmt(agg, "macro_f1")
        values["abl_{}_macro_mean".format(key)] = mean_of(agg, "macro_f1")
        values["abl_{}_attack".format(key)] = fmt(agg, "macro_f1_attacks")
        values["abl_{}_low".format(key)] = "{:.3f} ± {:.3f}".format(*low_means[key])
        values["abl_{}_low_mean".format(key)] = "{:.3f}".format(low_means[key][0])
        values["abl_{}_recon".format(key)] = "{:.3f}".format(agg["per_class"]["Reconnaissance"]["f1_mean"])
        for c in LOW_VOLUME:
            values["abl_{}_{}".format(key, c.lower())] = "{:.3f}".format(agg["per_class"][c]["f1_mean"])
    tables["ablation"] = {
        "caption": "Context-group ablation for gated CAG-IDS on the chronological split (mean over five seeds)",
        "colWidths": [3.0, 1.45, 1.45, 0.75, 0.75, 0.75, 0.75, 1.35, 0.75],
        "header": [["Context used", "Macro-F1", "Attack macro-F1", "Ana.", "Bkd.", "Shc.", "Wor.", "Low-volume mean", "Rec."]],
        "rows": rows,
        "midRules": [0, 4],
        "fontSize": 16,
        "align": ["left"] + ["center"] * 8,
        "notes": "Ana., Bkd., Shc. and Wor. give the mean F1 of Analysis, Backdoor, Shellcode and Worms, the low-volume classes named in H2; the low-volume mean averages them per seed (mean ± SD). Reconnaissance (Rec.) is reported separately. Every row uses the same architecture, training and seeds; only the context groups differ.",
    }
    drop = {k: low_means["full"][0] - low_means[k][0] for k in ("no_c1", "no_c2", "no_c3")}
    for k, v in drop.items():
        values["abl_drop_{}".format(k)] = signed(-v)
    values["h2_supported"] = "yes" if (drop["no_c2"] > max(drop["no_c1"], drop["no_c3"]) and
                                       drop["no_c2"] > max(low_means["full"][1], low_means["no_c2"][1])) else "no"

    # ---------------- gate analysis ----------------
    ctx_raw = np.load(work / "graphs" / "context_raw.npy")
    cn = meta["context_names"]
    degree = np.expm1(ctx_raw[:, cn.index("in_degree")]) + np.expm1(ctx_raw[:, cn.index("out_degree")])
    flows_df = pd.read_parquet(work / "flows.parquet", columns=["src_ip", "dst_ip"])
    names = host_names(work, flows_df, np.load(work / "graphs" / "row_index.npy"))
    is_att = np.isin(names, ATTACKERS)
    rhos, low_g, high_g, att_g, oth_g, means = [], [], [], [], [], []
    gate_curves = []
    for s in SEEDS:
        d = np.load(pred_dir / "cagids_seed{}.npz".format(s))
        g, nodes = d["gate_mean"], d["gate_nodes"]
        dd = degree[nodes]
        rhos.append(spearmanr(dd, g).correlation)
        low_g.append(g[dd <= 10].mean())
        high_g.append(g[dd >= 100].mean())
        att_g.append(g[is_att[nodes]].mean())
        oth_g.append(g[~is_att[nodes]].mean())
        means.append(g.mean())
        gate_curves.append((dd, g))
    values["gate_rho"] = "{:.2f} ± {:.2f}".format(np.mean(rhos), np.std(rhos, ddof=1))
    values["gate_rho_min"] = "{:.2f}".format(min(rhos))
    values["gate_rho_max"] = "{:.2f}".format(max(rhos))
    values["gate_mean"] = "{:.2f} ± {:.2f}".format(np.mean(means), np.std(means, ddof=1))
    values["gate_low_degree"] = "{:.2f} ± {:.2f}".format(np.mean(low_g), np.std(low_g, ddof=1))
    values["gate_high_degree"] = "{:.2f} ± {:.2f}".format(np.mean(high_g), np.std(high_g, ddof=1))
    values["gate_attackers"] = "{:.2f} ± {:.2f}".format(np.mean(att_g), np.std(att_g, ddof=1))
    values["gate_others"] = "{:.2f} ± {:.2f}".format(np.mean(oth_g), np.std(oth_g, ddof=1))
    values["gate_rho_sign_consistent"] = "yes" if (all(r > 0 for r in rhos) or all(r < 0 for r in rhos)) else "no"
    values["gate_rho_negative_seeds"] = num_word(sum(r < 0 for r in rhos))
    values["gate_rho_list"] = ", ".join(signed(r, 2) for r in rhos)
    values["gate_n_test_nodes"] = int_fmt(len(gate_curves[0][0]))
    dd0 = gate_curves[0][0]
    values["gate_share_low"] = pct_fmt((dd0 <= 10).mean(), 0)
    values["gate_share_high"] = pct_fmt((dd0 >= 100).mean(), 0)
    values["gate_low_higher_seeds"] = num_word(sum(l > h for l, h in zip(low_g, high_g)))

    # ---------------- split protocol ----------------
    proto = [("Random forest", "baseline_random_forest", "baseline_random_forest_random_flow", "random_forest", "random_forest_random_flow", "rf"),
             ("XGBoost", "baseline_xgboost", "baseline_xgboost_random_flow", "xgboost", "xgboost_random_flow", "xgb"),
             ("Flow-only MLP (B2)", "gnn_flow_mlp", "gnn_rf_flow_mlp", "flow_mlp", "rf_flow_mlp", "mlp"),
             ("E-GraphSAGE-style GNN (B3)", "gnn_egraphsage", "gnn_rf_egraphsage", "egraphsage", "rf_egraphsage", "b3"),
             ("Centrality-initialised GNN (B4)", "gnn_centrality_egraphsage", "gnn_rf_centrality_egraphsage", "centrality_egraphsage", "rf_centrality_egraphsage", "b4"),
             ("CAG-IDS, concatenation", "gnn_cagids_concat", "gnn_rf_cagids_concat", "cagids_concat", "rf_cagids_concat", "concat"),
             ("CAG-IDS, gated", "gnn_cagids", "gnn_rf_cagids", "cagids", "rf_cagids", "cagids")]
    rows, rows_same = [], []
    split_chrono = np.load(work / "graphs" / "edge_split.npy")
    split_random = np.load(work / "graphs_random_flow" / "edge_split.npy")
    if not np.array_equal(np.load(work / "graphs" / "row_index.npy"), np.load(work / "graphs_random_flow" / "row_index.npy")):
        raise SystemExit("the chronological and random-split graphs do not share their flow order")
    common = np.flatnonzero((split_chrono == 2) & (split_random == 2))
    values["same_flow_n"] = int_fmt(len(common))
    lab_all = np.load(work / "graphs" / "labels.npy")
    values["same_flow_attack"] = int_fmt((lab_all[common] > 0).sum())
    values["same_flow_windows"] = int_fmt(len(np.unique(edge_window[common])))
    gaps = {}
    same_share = {}
    for label, chrono, rnd, tag_c, tag_r, key in proto:
        aa, bb = agg_of(results, chrono), agg_of(results, rnd)
        gap = bb["macro_f1"]["mean"] - aa["macro_f1"]["mean"]
        gaps[key] = gap
        rows.append([label, fmt(bb, "macro_f1"), fmt(aa, "macro_f1"), signed(gap)])
        values["{}_random_macro".format(key)] = fmt(bb, "macro_f1")
        values["{}_random_attack".format(key)] = fmt(bb, "macro_f1_attacks")
        values["{}_gap".format(key)] = signed(gap)
        res = paired_bootstrap(P(tag_r), P(tag_c), edge_window, n_classes, keep_ids=common, n_boot=args.n_boot)
        ra = res["all"]
        rows_same.append([label, "{:.3f}".format(ra["a"]), "{:.3f}".format(ra["b"]), signed(ra["delta"]), interval(ra["lo"], ra["hi"])])
        values["same_{}".format(key)] = "{} {}".format(signed(ra["delta"]), interval(ra["lo"], ra["hi"]))
        values["same_{}_delta".format(key)] = signed(ra["delta"])
        values["same_{}_random".format(key)] = "{:.3f}".format(ra["a"])
        values["same_{}_chrono".format(key)] = "{:.3f}".format(ra["b"])
        values["same_{}_lo".format(key)] = signed(ra["lo"])
        same_share[key] = ra["delta"] / gap if gap > 0 else float("nan")
    tables["protocol"] = {
        "caption": "Macro-F1 under a random flow split and under the chronological split, each on its own test set",
        "colWidths": [3.6, 2.0, 2.0, 1.2],
        "header": [["Model", "Random flow split", "Chronological split", "Gap"]],
        "rows": rows,
        "midRules": [1],
        "fontSize": 17,
        "align": ["left", "center", "center", "center"],
        "notes": "The random split assigns flows, stratified by class, to training, validation and test (70/10/20), while every window graph keeps all of its flows. Context is computed in the same window-causal way in both cases. The two protocols are scored on different test sets; Table [#tab:sameflow] removes this difference.",
    }
    tables["sameflow"] = {
        "caption": "Random-split and chronological models scored on the same flows",
        "colWidths": [3.6, 1.6, 1.6, 1.1, 1.9],
        "header": [["Model", "Trained on random split", "Trained chronologically", "Difference", "95% interval"]],
        "rows": rows_same,
        "midRules": [1],
        "fontSize": 17,
        "align": ["left", "center", "center", "center", "center"],
        "notes": "Macro-F1 (seed mean) on the {} flows that belong to both test sets, that is, the random-split test flows that fall in the chronological test period ({} attack flows in {} windows). Intervals come from the moving-block bootstrap over those windows.".format(
            values["same_flow_n"], values["same_flow_attack"], values["same_flow_windows"]),
    }
    shares = [v for v in same_share.values() if np.isfinite(v)]
    values["same_share_min"] = pct_fmt(min(shares), 0)
    values["same_share_max"] = pct_fmt(max(shares), 0)
    gnn_gaps = [gaps[k] for k in ("b3", "b4", "concat", "cagids")]
    values["gap_gnn_min"] = signed(min(gnn_gaps))
    values["gap_gnn_max"] = signed(max(gnn_gaps))
    values["gap_flat_min"] = signed(min(gaps["rf"], gaps["xgb"]))
    values["gap_flat_max"] = signed(max(gaps["rf"], gaps["xgb"]))
    rfr = agg_of(results, "baseline_random_forest_random_flow")
    values["rnd_test_analysis"] = int_fmt(rfr["per_class"]["Analysis"]["support"])
    values["rnd_test_attack"] = int_fmt(sum(v["support"] for k, v in rfr["per_class"].items() if k != "Benign"))
    values["rf_backdoor_f1_random"] = "{:.3f}".format(rfr["per_class"]["Backdoor"]["f1_mean"])

    # ---------------- context probe tests (why context hurts the forest) ----------------
    ct = load(results, "context_probe_tests")
    order = [("chronological/flow_only", "Flow features only", "Chronological"),
             ("chronological/flow_plus_context", "Flow features + endpoint context", "Chronological"),
             ("chronological/flow_plus_shuffled_context", "Flow features + shuffled context", "Chronological"),
             ("chronological/flow_plus_context_without_c2", "Flow features + context without C2", "Chronological"),
             ("chronological/flow_plus_c2_only", "Flow features + C2 only", "Chronological"),
             ("random_flow/flow_only", "Flow features only", "Random flow"),
             ("random_flow/flow_plus_context", "Flow features + endpoint context", "Random flow")]
    rows = []
    for key, label, split in order:
        r = ct[key]
        share = "{:.2f}".format(r["context_importance_share"]["mean"]) if "context_importance_share" in r else "–"
        rows.append([label, split, "{:.3f} ± {:.3f}".format(r["macro_f1"]["mean"], r["macro_f1"]["std"]),
                     "{:.3f}".format(r["macro_f1_attacks"]["mean"]), share])
        k = key.replace("/", "_")
        values["ct_{}".format(k)] = "{:.3f} ± {:.3f}".format(r["macro_f1"]["mean"], r["macro_f1"]["std"])
        values["ct_{}_mean".format(k)] = "{:.3f}".format(r["macro_f1"]["mean"])
        values["ct_{}_attack".format(k)] = "{:.3f}".format(r["macro_f1_attacks"]["mean"])
        if "context_importance_share" in r:
            values["ct_{}_share".format(k)] = "{:.2f}".format(r["context_importance_share"]["mean"])
    c0 = ct["chronological/flow_only"]["macro_f1"]["mean"]
    for k in ("flow_plus_context", "flow_plus_shuffled_context", "flow_plus_context_without_c2", "flow_plus_c2_only"):
        values["ct_delta_{}".format(k)] = signed(ct["chronological/" + k]["macro_f1"]["mean"] - c0)
    r0 = ct["random_flow/flow_only"]["macro_f1"]["mean"]
    values["ct_delta_random_context"] = signed(ct["random_flow/flow_plus_context"]["macro_f1"]["mean"] - r0)
    values["ct_backdoor_flow"] = "{:.3f}".format(ct["chronological/flow_only"]["per_class_f1"]["Backdoor"])
    values["ct_backdoor_context"] = "{:.3f}".format(ct["chronological/flow_plus_context"]["per_class_f1"]["Backdoor"])
    values["ct_random_backdoor_flow"] = "{:.3f}".format(ct["random_flow/flow_only"]["per_class_f1"]["Backdoor"])
    values["ct_random_backdoor_context"] = "{:.3f}".format(ct["random_flow/flow_plus_context"]["per_class_f1"]["Backdoor"])
    tables["contexttests"] = {
        "caption": "Random-forest probes that separate the explanations for the effect of endpoint context (mean over three seeds)",
        "colWidths": [3.9, 1.6, 1.6, 1.4, 1.5],
        "header": [["Inputs", "Split", "Macro-F1", "Attack macro-F1", "Context share of importance"]],
        "rows": rows,
        "midRules": [4],
        "fontSize": 17,
        "align": ["left", "left", "center", "center", "center"],
        "notes": "Same forest as the context probes in Table [#tab:diagnostics] (200 trees, 400,000-flow subsample, seeds 1 to 3). Shuffled context permutes the rows of the 34 context columns, which keeps their distribution but removes any link to the flow or its label. The last column is the share of the forest's impurity-based importance that falls on context columns.",
    }

    # ---------------- sensitivity (window length, post-hoc training settings) ----------------
    win = {}
    for tag in ("graphs_w30", "graphs_w300"):
        m = json.load(open(work / tag / "meta.json", encoding="utf-8"))
        win[tag] = m["n_windows"]
        values["n_windows_{}".format(tag.split("_")[1])] = int_fmt(m["n_windows"])
    settings = [("Window 30 s ({} windows)".format(int_fmt(win["graphs_w30"])), "w30_{}"),
                ("Window 60 s, planned settings ({} windows)".format(int_fmt(meta["n_windows"])), "{}"),
                ("Window 300 s ({} windows)".format(int_fmt(win["graphs_w300"])), "w300_{}"),
                ("√-inverse class weights", "sw_{}"),
                ("Focal loss (*γ* = 2)", "focal_{}"),
                ("60 epochs, patience 12", "long_{}")]
    rows = []
    for label, pattern in settings:
        row = [label]
        aggs = []
        for tag, key, _ in NEURAL:
            name = pattern.format(tag)
            if pattern == "focal_{}" and tag == "cagids":
                name = "cagids_focal"
            agg = agg_of(results, "gnn_" + name)
            aggs.append(agg)
            row.append(fmt(agg, "macro_f1"))
            sk = pattern.split("_")[0] if "_" in pattern else "planned"
            values["sens_{}_{}".format(sk, key)] = fmt(agg, "macro_f1")
            values["sens_{}_{}_mean".format(sk, key)] = mean_of(agg, "macro_f1")
            values["sens_{}_{}_attack".format(sk, key)] = fmt(agg, "macro_f1_attacks")
        best = max(a["macro_f1"]["mean"] for a in aggs)
        row = [row[0]] + [{"text": c, "bold": a["macro_f1"]["mean"] == best} for c, a in zip(row[1:], aggs)]
        rows.append(row)
    tables["sensitivity"] = {
        "caption": "Macro-F1 of the neural models under other window lengths and, post hoc, other training settings (chronological split, mean ± SD over five seeds)",
        "colWidths": [3.2, 1.5, 1.5, 1.5, 1.5, 1.5],
        "header": [["Setting"] + [lab for _, _, lab in NEURAL]],
        "rows": rows,
        "midRules": [2],
        "fontSize": 16,
        "align": ["left"] + ["center"] * 5,
        "notes": "Each window length uses its own 70/10/20 chronological split of its windows. The last three rows change one training setting at a time. Apart from focal loss for gated CAG-IDS, which was planned, they were added after the planned runs had shown that the neural models over-predict the rare classes; the planned settings remain the primary results. The random forest reaches {} for comparison. Bold: best in the row.".format(values["m_rf_macro"]),
    }
    sens = {}
    for _, p in settings:
        sk = p.split("_")[0] if "_" in p else "planned"
        for t, key, _ in NEURAL:
            name = "cagids_focal" if (p == "focal_{}" and t == "cagids") else p.format(t)
            sens[(sk, key)] = agg_of(results, "gnn_" + name)["macro_f1"]["mean"]
    setting_keys = sorted({k for k, _ in sens})
    values["sens_best_overall"] = "{:.3f}".format(max(sens.values()))
    best_key = max(sens, key=sens.get)
    values["sens_best_overall_model"] = {k: lab for _, k, lab in NEURAL}[best_key[1]]
    for sk in setting_keys:
        if sk == "planned":
            continue
        gains = [sens[(sk, k)] - sens[("planned", k)] for _, k, _ in NEURAL]
        values["sens_{}_gain_min".format(sk)] = signed(min(gains))
        values["sens_{}_gain_max".format(sk)] = signed(max(gains))
    rows_mlp_best = sum(max((sens[(sk, k)], k) for _, k, _ in NEURAL)[1] == "mlp" for sk in setting_keys)
    values["sens_rows_mlp_best"] = num_word(rows_mlp_best)
    values["sens_rows_total"] = num_word(len(setting_keys))
    values["sens_rows_cagids_above_mlp"] = num_word(sum(sens[(sk, "cagids")] > sens[(sk, "mlp")] for sk in setting_keys))
    values["sens_rows_cagids_above_b3"] = num_word(sum(sens[(sk, "cagids")] > sens[(sk, "b3")] for sk in setting_keys))
    values["sens_rows_b4_above_cagids"] = num_word(sum(sens[(sk, "b4")] > sens[(sk, "cagids")] for sk in setting_keys))
    values["sens_rows_graph_above_mlp"] = num_word(sum(any(sens[(sk, k)] > sens[(sk, "mlp")] for k in ("b3", "b4", "concat", "cagids"))
                                                       for sk in setting_keys))
    values["sens_max_gap_to_rf"] = "{:.3f}".format(agg_of(results, "baseline_random_forest")["macro_f1"]["mean"] - max(sens.values()))

    # ---------------- efficiency ----------------
    eff_rows = [["Graph and context construction (CPU, one process)", "{:.2f} ms per 60-s window".format(meta["ms_per_window_build"])]]
    for name, label, key in [("gnn_flow_mlp", "Flow-only MLP (B2)", "mlp"), ("gnn_egraphsage", "E-GraphSAGE-style GNN (B3)", "b3"),
                             ("gnn_cagids", "Gated CAG-IDS", "cagids"), ("gnn_cagids_gatv2", "Gated CAG-IDS, GATv2", "gatv2")]:
        p = load(results, name)
        inf = np.mean([r["inference"]["ms_per_window_mean"] for r in p["runs"]])
        tr = np.mean([r["train_seconds"] for r in p["runs"]])
        eff_rows.append(["{}: forward pass for one window (GPU)".format(label), "{:.2f} ms".format(inf)])
        eff_rows.append(["{}: training run (GPU, up to 30 epochs)".format(label), "{:.0f} s".format(tr)])
        values["infer_ms_{}".format(key)] = "{:.2f}".format(inf)
        values["train_s_{}".format(key)] = "{:.0f}".format(tr)
    for name, label, key in [("baseline_random_forest", "Random forest", "rf"), ("baseline_xgboost", "XGBoost", "xgb")]:
        p = load(results, name)
        fps = np.mean([r["inference"]["flows_per_second"] for r in p["runs"]])
        tr = np.mean([r["train_seconds"] for r in p["runs"]])
        eff_rows.append(["{}: training (CPU, all cores, 400,000 flows)".format(label), "{:.1f} s".format(tr)])
        eff_rows.append(["{}: batch prediction (CPU, all cores)".format(label), "{:,.0f} flows per second".format(fps)])
        values["{}_flows_per_second".format(key)] = "{:,.0f}".format(fps)
        values["{}_train_s".format(key)] = "{:.1f}".format(tr)
    tables["efficiency"] = {
        "caption": "Measured processing cost",
        "colWidths": [6.0, 3.0],
        "header": [["Stage", "Cost"]],
        "rows": eff_rows,
        "midRules": [0, 8],
        "fontSize": 17,
        "align": ["left", "left"],
        "notes": "AMD Ryzen 7 7840HS CPU and NVIDIA GeForce RTX 4060 Laptop GPU. Forward passes are timed one window at a time over up to 200 test windows after five warm-up windows, as a deployed detector would run; training times are means over the five seeds. A 60-s window holds {} flows at the median.".format(values["flows_per_window_median"]),
    }

    # ---------------- experiment inventory ----------------
    inventory = [
        ("Dataset audit and probes", "Shortcuts, host overlap, context information", "1", "sec:audit"),
        ("RF and XGBoost, chronological split", "Flat baselines (B1)", "2 × 5 seeds", "sec:main"),
        ("B2, B3, B4, CAG-IDS concat., gated, GATv2", "Main comparison (H1, H3)", "6 × 5 seeds", "sec:main"),
        ("C1 + gate control", "Context groups against the direct context path", "1 × 5 seeds", "sec:ablation"),
        ("Neural context-only probe", "Context information without topology", "1 × 5 seeds", "sec:main"),
        ("Context-group ablations", "Contribution of C1, C2, C3; C4 check (H2)", "4 × 5 seeds", "sec:ablation"),
        ("RF, XGBoost and five neural models, random flow split", "Effect of the split protocol (RQ2)", "2 × 5 + 5 × 5 seeds", "sec:protocol"),
        ("Forest probes with shuffled and partial context", "Why context hurts a flat classifier", "7 × 3 seeds", "sec:contexttests"),
        ("Window length 30 s and 300 s", "Sensitivity to the window", "10 × 5 seeds", "sec:sensitivity"),
        ("√-inverse weights, focal loss, longer training (post hoc)", "Sensitivity to the training settings", "15 × 5 seeds", "sec:sensitivity"),
    ]
    tables["inventory"] = {
        "caption": "Experiments of the study",
        "colWidths": [3.8, 3.4, 1.6, 1.2],
        "header": [["Experiment", "Purpose", "Runs", "Results"]],
        "rows": [[a, b, c, "[sec:{}]".format(d.split(":")[1])] for a, b, c, d in inventory],
        "fontSize": 17,
        "align": ["left", "left", "center", "center"],
        "notes": "All runs are started by scripts/run_all.py, and scripts/analyze.py produces every results table, data figure and quoted number from their outputs. The neural models account for {} training runs.".format(
            int_fmt(5 * (6 + 1 + 1 + 4 + 5 + 10 + 15))),
    }
    values["n_neural_runs"] = int_fmt(5 * (6 + 1 + 1 + 4 + 5 + 10 + 15))
    co = agg_of(results, "gnn_context_only")
    values["m_ctxonly_macro"] = fmt(co, "macro_f1")
    values["m_ctxonly_attack"] = fmt(co, "macro_f1_attacks")
    values["m_ctxonly_binary"] = fmt(co, "binary_f1")
    c1g = agg_of(results, "gnn_c1_gate")
    values["m_c1gate_macro"] = fmt(c1g, "macro_f1")
    values["m_c1gate_attack"] = fmt(c1g, "macro_f1_attacks")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tables_generated.json", "w", encoding="utf-8") as handle:
        json.dump(tables, handle, indent=1, ensure_ascii=False)
    with open(out_dir / "values.json", "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=1, ensure_ascii=False)
    print("wrote", out_dir / "tables_generated.json", "and values.json with", len(values), "values")

    make_figures(work, results, figdir, classes, diag, meta, gate_curves)


# ----------------------------------------------------------------------------

def make_figures(work, results, figdir, classes, diag, meta, gate_curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.fontset": "stix"})
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
    colors = dict(zip([c for c in classes if c != "Benign"], palette))

    # timeline -------------------------------------------------------------
    tl = pd.read_csv(results / "timeline_graphs.csv", index_col=0)
    t = pd.to_datetime(tl.index, unit="ms", utc=True)
    sessions = [(t < pd.Timestamp("2015-02-01", tz="UTC")), (t >= pd.Timestamp("2015-02-01", tz="UTC"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})
    for ax, sel, title in zip(axes, sessions, ["Session 1 (22 to 23 Jan 2015, UTC)", "Session 2 (18 Feb 2015, UTC)"]):
        part = tl[sel]
        tt = t[sel]
        hours = (tt - tt[0]).total_seconds() / 3600.0 + tt[0].hour + tt[0].minute / 60.0
        ax.plot(hours, part["Benign"].values, color="#999999", lw=0.8, label="Benign")
        bottom = np.zeros(len(part))
        for c in [c for c in classes if c != "Benign" and c in part.columns]:
            ax.bar(hours, part[c].values, bottom=bottom, width=1 / 6.0, color=colors[c], label=c, align="edge", linewidth=0)
            bottom += part[c].values
        ax.set_yscale("symlog", linthresh=10)
        split = part["split"].values
        for code, shade, name in [(1, "#fde0c5", "validation"), (2, "#d9e8f5", "test")]:
            m = split == code
            if m.any():
                ax.axvspan(hours[m].min(), hours[m].max() + 1 / 6.0, color=shade, zorder=0, lw=0)
                ax.text((hours[m].min() + hours[m].max()) / 2, 2.2e4, name, ha="center", fontsize=6.5)
        ax.set_title(title)
        ax.set_xlabel("hour of day (UTC)")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("flows per 10 min (symlog)")
    axes[0].set_ylim(0, 6e4)
    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=10, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.0, columnspacing=0.8)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.9, bottom=0.3)
    fig.savefig(figdir / "fig2_timeline.png", dpi=300)
    plt.close(fig)

    # single-feature probes ---------------------------------------------------
    sf = diag["single_feature_probes"]
    items = sorted(sf.items(), key=lambda kv: -kv[1]["binary_f1"])
    kept = [kv for kv in items if not kv[0].startswith("probe__")][:12]
    excluded = [kv for kv in items if kv[0].startswith("probe__")]
    chosen = sorted(excluded + kept, key=lambda kv: kv[1]["binary_f1"])
    names = [k.replace("probe__", "") + (" *" if k.startswith("probe__") else "") for k, _ in chosen]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    y = np.arange(len(chosen))
    ax.barh(y - 0.2, [v["binary_f1"] for _, v in chosen], height=0.38, color="#e15759", label="binary F1")
    ax.barh(y + 0.2, [v["macro_f1"] for _, v in chosen], height=0.38, color="#4e79a7", label="ten-class macro-F1")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test score of a depth-6 tree trained on one feature")
    ax.grid(axis="x", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_single_feature.png", dpi=300)
    plt.close(fig)

    # context distributions of attacker hosts and other hosts ---------------------
    ctx = np.load(work / "graphs" / "context_raw.npy")
    flows_df = pd.read_parquet(work / "flows.parquet", columns=["src_ip", "dst_ip"])
    names_all = host_names(work, flows_df, np.load(work / "graphs" / "row_index.npy"))
    is_att = np.isin(names_all, ATTACKERS)
    cn = meta["context_names"]
    feats = [("out_degree", "log(1 + out-degree)"), ("dst_port_entropy", "destination-port entropy (bits)"),
             ("peer_novelty", "peer novelty ratio"), ("z_unique_ports", "z-score of distinct destination ports")]
    fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.2))
    for ax, (f, lab) in zip(axes.ravel(), feats):
        j = cn.index(f)
        a, b = ctx[is_att, j], ctx[~is_att, j]
        lo, hi = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        ax.hist(b, bins=bins, density=True, alpha=0.75, color="#76b7b2", label="other hosts")
        ax.hist(a, bins=bins, density=True, alpha=0.7, color="#e15759", label="4 attack sources")
        ax.set_xlabel(lab)
        ax.set_yticks([])
    axes.ravel()[0].legend(frameon=False, fontsize=6.5, loc="upper center")
    fig.tight_layout(pad=0.4)
    fig.savefig(figdir / "fig4_context_separation.png", dpi=300)
    plt.close(fig)

    # gate value against host degree ------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    all_deg = np.concatenate([d for d, _ in gate_curves])
    edges = np.unique(np.round(np.logspace(0, np.log10(max(all_deg.max(), 10)) + 1e-9, 15)))
    for k, (dd, g) in enumerate(gate_curves):
        idx = np.digitize(dd, edges)
        xs, ys = [], []
        for b in np.unique(idx):
            m = idx == b
            if m.sum() >= 20:
                xs.append(np.median(dd[m]))
                ys.append(g[m].mean())
        ax.plot(xs, ys, marker="o", ms=2.5, lw=0.9, alpha=0.85, label="seed {}".format(k + 1))
    ax.set_xscale("log")
    ax.set_xlabel("flows of the host in the window (log scale)")
    ax.set_ylabel("mean gate value (weight on topology)")
    ax.grid(color="#e6e6e6", lw=0.5)
    ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(figdir / "fig5_gate_degree.png", dpi=300)
    plt.close(fig)
    print("figures written to", figdir)


if __name__ == "__main__":
    try:
        main()
    except Missing as exc:
        raise SystemExit("result file missing: {}. Run scripts/run_all.py first.".format(exc))
