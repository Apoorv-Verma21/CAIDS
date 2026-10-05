import re

import numpy as np
import pandas as pd


CANONICAL = {
    "src_ip": ["IPV4_SRC_ADDR", "IPV6_SRC_ADDR", "SRC_IP", "SRCIP", "SOURCE_IP", "SA"],
    "dst_ip": ["IPV4_DST_ADDR", "IPV6_DST_ADDR", "DST_IP", "DSTIP", "DEST_IP", "DA"],
    "src_port": ["L4_SRC_PORT", "SRC_PORT", "SPORT", "SOURCE_PORT"],
    "dst_port": ["L4_DST_PORT", "DST_PORT", "DPORT", "DEST_PORT"],
    "protocol": ["PROTOCOL", "PROTO", "IP_PROTOCOL_VERSION"],
    "start_time": [
        "FLOW_START_MILLISECONDS",
        "FLOW_START_MS",
        "FLOW_START",
        "TIMESTAMP",
        "STIME",
        "FIRST_SWITCHED",
    ],
    "end_time": ["FLOW_END_MILLISECONDS", "FLOW_END_MS", "FLOW_END", "LTIME", "LAST_SWITCHED"],
    "label": ["LABEL", "BINARY_LABEL", "IS_ATTACK"],
    "attack": ["ATTACK", "ATTACK_CAT", "ATTACK_CATEGORY", "CATEGORY", "CLASS"],
    "in_bytes": ["IN_BYTES", "SRC_TO_DST_BYTES"],
    "out_bytes": ["OUT_BYTES", "DST_TO_SRC_BYTES"],
    "in_pkts": ["IN_PKTS", "SRC_TO_DST_PKTS"],
    "out_pkts": ["OUT_PKTS", "DST_TO_SRC_PKTS"],
}

IDENTIFIER_ROLES = ("src_ip", "dst_ip", "src_port", "dst_port", "start_time", "end_time")
TARGET_ROLES = ("label", "attack")
REQUIRED_ROLES = ("src_ip", "dst_ip", "start_time", "label")


def _normalise(name):
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def resolve_columns(columns):
    lookup = {}
    for col in columns:
        lookup.setdefault(_normalise(col), col)

    resolved = {}
    for role, candidates in CANONICAL.items():
        for candidate in candidates:
            key = _normalise(candidate)
            if key in lookup:
                resolved[role] = lookup[key]
                break
    return resolved


def describe_resolution(columns):
    resolved = resolve_columns(columns)
    lines = ["Column resolution:"]
    for role in CANONICAL:
        lines.append("  {:<12} -> {}".format(role, resolved.get(role, "NOT FOUND")))
    missing = [r for r in REQUIRED_ROLES if r not in resolved]
    if missing:
        lines.append("  MISSING REQUIRED: " + ", ".join(missing))
    return "\n".join(lines), resolved


def load_flows(path, nrows=None, usecols=None):
    df = pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_milliseconds(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().mean() > 0.9:
        values = values.astype("float64")
        median = float(np.nanmedian(values.values))
        if median > 1e17:
            values = values / 1e6
        elif median > 1e14:
            values = values / 1e3
        elif 0 < median < 1e11:
            values = values * 1000.0
        return values
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.astype("int64") / 1e6


def prepare_frame(df, cfg):
    resolved = resolve_columns(df.columns)
    missing = [r for r in REQUIRED_ROLES if r not in resolved]
    if missing:
        raise ValueError(
            "Could not find required columns for roles: {}. Available columns: {}".format(
                ", ".join(missing), list(df.columns)
            )
        )

    out = pd.DataFrame(index=df.index)
    out["src_ip"] = df[resolved["src_ip"]].astype(str)
    out["dst_ip"] = df[resolved["dst_ip"]].astype(str)
    out["src_port"] = pd.to_numeric(df[resolved["src_port"]], errors="coerce").fillna(0).astype("int32") if "src_port" in resolved else 0
    out["dst_port"] = pd.to_numeric(df[resolved["dst_port"]], errors="coerce").fillna(0).astype("int32") if "dst_port" in resolved else 0
    out["start_ms"] = _to_milliseconds(df[resolved["start_time"]])

    label_raw = df[resolved["label"]]
    if label_raw.dtype == object:
        out["label"] = (
            label_raw.astype(str).str.strip().str.lower().isin(["1", "true", "attack", "malicious", "yes"]).astype("int8")
        )
    else:
        out["label"] = (pd.to_numeric(label_raw, errors="coerce").fillna(0) > 0).astype("int8")

    if "attack" in resolved:
        attack = df[resolved["attack"]].astype(str).str.strip()
        attack = attack.replace({"": "Benign", "nan": "Benign", "None": "Benign", "-": "Benign"})
        attack = attack.where(out["label"].values == 1, "Benign")
        out["attack"] = attack
    else:
        out["attack"] = np.where(out["label"].values == 1, "Attack", "Benign")

    identifier_cols = {resolved[r] for r in IDENTIFIER_ROLES if r in resolved}
    target_cols = {resolved[r] for r in TARGET_ROLES if r in resolved}
    drop = identifier_cols | target_cols
    if cfg.data.get("drop_leaky", True):
        leaky = {_normalise(c) for c in cfg.data.get("leaky_features", [])}
        drop |= {c for c in df.columns if _normalise(c) in leaky}

    feature_cols = []
    for col in df.columns:
        if col in drop:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() < 0.5:
            continue
        feature_cols.append(col)

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    if "protocol" in resolved and resolved["protocol"] in feature_cols:
        proto = pd.to_numeric(df[resolved["protocol"]], errors="coerce").fillna(0).astype(int)
        for value, name in [(6, "tcp"), (17, "udp"), (1, "icmp")]:
            features["proto_is_{}".format(name)] = (proto == value).astype("float32")

    out = pd.concat([out, features], axis=1)
    out = out.dropna(subset=["start_ms"])
    out = out.sort_values("start_ms", kind="mergesort").reset_index(drop=True)

    meta = {
        "resolved": resolved,
        "feature_cols": list(features.columns),
        "dropped_cols": sorted(drop),
    }
    return out, meta


def assign_windows(df, window_seconds):
    start = df["start_ms"].min()
    width = float(window_seconds) * 1000.0
    window_id = ((df["start_ms"] - start) // width).astype("int64")
    df = df.copy()
    df["window_id"] = window_id
    return df


def chronological_split(window_ids, cfg):
    unique = np.unique(window_ids)
    n = len(unique)
    mode = cfg.split.get("mode", "chronological")

    if mode == "blocked":
        requested = int(cfg.split.get("blocks", 10))
        blocks = max(1, min(requested, n // 3)) if n >= 3 else 1
        assignment = {}
        per_block = max(1, n // blocks)
        for b in range(blocks):
            lo = b * per_block
            hi = n if b == blocks - 1 else (b + 1) * per_block
            chunk = unique[lo:hi]
            size = len(chunk)
            if size == 0:
                continue
            if size >= 3:
                n_va = max(1, int(size * cfg.split["val"]))
                n_te = max(1, int(size * cfg.split["test"]))
                n_tr = max(1, size - n_va - n_te)
                n_va = min(n_va, size - n_tr)
                n_te = size - n_tr - n_va
            else:
                n_tr, n_va, n_te = size, 0, 0
            for w in chunk[:n_tr]:
                assignment[w] = "train"
            for w in chunk[n_tr:n_tr + n_va]:
                assignment[w] = "val"
            for w in chunk[n_tr + n_va:]:
                assignment[w] = "test"
        return assignment

    n_tr = int(n * cfg.split["train"])
    n_va = int(n * cfg.split["val"])
    assignment = {}
    for w in unique[:n_tr]:
        assignment[w] = "train"
    for w in unique[n_tr:n_tr + n_va]:
        assignment[w] = "val"
    for w in unique[n_tr + n_va:]:
        assignment[w] = "test"
    return assignment


def standardise(train_matrix, matrices):
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return [((m - mean) / std).astype("float32") for m in matrices]


def log_transform(matrix, columns, all_columns):
    matrix = matrix.copy()
    for col in columns:
        if col in all_columns:
            idx = all_columns.index(col)
            matrix[:, idx] = np.log1p(np.clip(matrix[:, idx], 0, None))
    return matrix
