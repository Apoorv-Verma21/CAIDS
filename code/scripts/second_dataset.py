"""Identity audit and the three splitting protocols with the flat baselines on a second NetFlow v3 dataset.

    python scripts/second_dataset.py --root D:/CAIDS_ton --csv data/NF-ToN-IoT-v3.csv --out results/second_dataset.json

The CSV is read in chunks, so datasets with tens of millions of flows fit in memory. The features follow
prepare_frame (identifiers, targets and the four excluded fields removed, protocol indicators added), and the flat
baselines follow Section 5.4: all training attack flows plus benign flows up to 400,000 (if the attack flows alone
exceed that, each class is capped so that rare classes are kept whole), 200-tree random forest and XGBoost, seeds 1 to 5.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.data import resolve_columns, _normalise, _to_milliseconds  # noqa: E402

LEAKY = ["MIN_TTL", "MAX_TTL", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]
IDENT = ("src_ip", "dst_ip", "src_port", "dst_port", "start_time", "end_time")
SPLITS = ("chronological", "random_flow", "random_window")
ALL_SPLITS = ("chronological", "blocked", "random_flow", "random_window")


def read_chunks(path, usecols=None, chunksize=1_000_000):
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        yield chunk


def labels_of(chunk, resolved):
    raw = chunk[resolved["label"]]
    if raw.dtype == object:
        label = raw.astype(str).str.strip().str.lower().isin(["1", "true", "attack", "malicious", "yes"]).to_numpy()
    else:
        label = (pd.to_numeric(raw, errors="coerce").fillna(0) > 0).to_numpy()
    attack = chunk[resolved["attack"]].astype(str).str.strip()
    attack = attack.replace({"": "Benign", "nan": "Benign", "None": "Benign", "-": "Benign"}).to_numpy(object)
    attack[~label] = "Benign"
    return label.astype("int8"), attack


def chunk_features(chunk, feature_cols, proto_col):
    x = chunk.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32").to_numpy()
    if proto_col is not None:
        proto = pd.to_numeric(chunk[proto_col], errors="coerce").fillna(0).astype(int).to_numpy()
        x = np.hstack([x] + [(proto == v).astype("float32")[:, None] for v in (6, 17, 1)])
    return x


def macro_f1(y, p, n_classes, keep=None):
    present = np.bincount(y, minlength=n_classes) > 0
    if keep is not None:
        present &= keep
    tp = np.bincount(y[y == p], minlength=n_classes).astype(float)
    pc = np.bincount(p, minlength=n_classes).astype(float)
    sc = np.bincount(y, minlength=n_classes).astype(float)
    f1 = np.where(pc + sc > 0, 2 * tp / np.maximum(pc + sc, 1e-12), 0.0)
    return float(f1[present].mean())


def binary_f1(y, p):
    yt, yp = y != 0, p != 0
    tp = float((yt & yp).sum())
    return 2 * tp / max(yt.sum() + yp.sum(), 1)


def window_counts(y, p, win, n_classes):
    """Per-window counts of true positives, predictions and support, for the block bootstrap."""
    wins, inv = np.unique(win, return_inverse=True)
    tp = np.zeros((len(wins), n_classes))
    pc = np.zeros((len(wins), n_classes))
    sc = np.zeros((len(wins), n_classes))
    np.add.at(tp, (inv[y == p], y[y == p]), 1)
    np.add.at(pc, (inv, p), 1)
    np.add.at(sc, (inv, y), 1)
    return tp, pc, sc


def macro_from_counts(tp, pc, sc, keep=None):
    tp, pc, sc = tp.sum(0), pc.sum(0), sc.sum(0)
    present = sc > 0
    if keep is not None:
        present &= keep
    f1 = np.where(pc + sc > 0, 2 * tp / np.maximum(pc + sc, 1e-12), 0.0)
    return float(f1[present].mean())


def paired_gain(counts_a, counts_b, n_boot=2000, block=10, seed=0, keep=None):
    """Two-level bootstrap of mean(A seeds) - mean(B seeds): blocks of windows shared by both, seeds resampled.
    keep restricts the macro average to some classes (e.g. those present in both training sets)."""
    rng = np.random.default_rng(seed)
    n = counts_a[0][0].shape[0]
    starts_all = np.arange(max(n - block + 1, 1))
    nb = int(np.ceil(n / block))
    point = (np.mean([macro_from_counts(*c, keep=keep) for c in counts_a])
             - np.mean([macro_from_counts(*c, keep=keep) for c in counts_b]))
    reps = []
    for _ in range(n_boot):
        idx = (rng.choice(starts_all, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        fa = [macro_from_counts(*(m[idx] for m in c), keep=keep) for c in counts_a]
        fb = [macro_from_counts(*(m[idx] for m in c), keep=keep) for c in counts_b]
        reps.append(np.mean(rng.choice(fa, len(fa))) - np.mean(rng.choice(fb, len(fb))))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/second_dataset.json")
    ap.add_argument("--window-seconds", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--subsample", type=int, default=400_000)
    ap.add_argument("--models", nargs="*", default=["random_forest", "xgboost"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS), choices=ALL_SPLITS)
    ap.add_argument("--base", default="chronological", help="split whose test flows the random splits are compared on")
    ap.add_argument("--audit-only", action="store_true", help="write the identity audit on the --base split and stop")
    args = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(args.root)
    csv = str(root / args.csv)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv, nrows=1000)
    header.columns = [str(c).strip() for c in header.columns]
    resolved = resolve_columns(header.columns)
    for role in ("src_ip", "dst_ip", "start_time", "label", "attack"):
        assert role in resolved, "column for {} not found: {}".format(role, list(header.columns))
    drop = {resolved[r] for r in IDENT if r in resolved} | {resolved["label"], resolved["attack"]}
    leaky = {_normalise(c) for c in LEAKY}
    drop |= {c for c in header.columns if _normalise(c) in leaky}
    feature_cols = [c for c in header.columns if c not in drop
                    and pd.to_numeric(header[c], errors="coerce").notna().mean() >= 0.5]
    proto_col = resolved.get("protocol") if resolved.get("protocol") in feature_cols else None
    print("features:", len(feature_cols), "+ protocol indicators" if proto_col else "", flush=True)

    # ------------------------------------------------------------ pass 1: addresses, time and labels of every flow
    cols = [resolved[r] for r in ("src_ip", "dst_ip", "start_time", "label", "attack")]
    # addresses and class names are kept as integer codes, so tens of millions of flows fit in memory
    ip_code, cls_code = {}, {}

    def codes(values, table):
        for v in pd.unique(values):
            if v not in table:
                table[v] = len(table)
        return pd.Series(values).map(table).to_numpy("int32")

    src, dst, start, lab, att = [], [], [], [], []
    for chunk in read_chunks(csv, usecols=cols):
        src.append(codes(chunk[resolved["src_ip"]].astype(str).to_numpy(object), ip_code))
        dst.append(codes(chunk[resolved["dst_ip"]].astype(str).to_numpy(object), ip_code))
        start.append(_to_milliseconds(chunk[resolved["start_time"]]).to_numpy("float64"))
        l_, a_ = labels_of(chunk, resolved)
        lab.append(l_)
        att.append(codes(a_, cls_code))
    src, dst, start = np.concatenate(src), np.concatenate(dst), np.concatenate(start)
    lab, att = np.concatenate(lab), np.concatenate(att)
    ip_names = np.array(sorted(ip_code, key=ip_code.get), dtype=object)
    n = len(lab)
    ok = np.isfinite(start)
    cls_names = sorted(cls_code, key=cls_code.get)
    classes = ["Benign"] + sorted(c for c in cls_names if c != "Benign")
    remap = np.array([classes.index(c) for c in cls_names], dtype="int16")
    y = remap[att]
    del att
    n_classes = len(classes)
    win = np.full(n, -1, dtype="int64")
    win[ok] = ((start[ok] - start[ok].min()) // (args.window_seconds * 1000.0)).astype("int64")
    wcount = np.bincount(win[ok])
    valid = ok & (wcount[np.clip(win, 0, None)] >= 2)
    print("flows {:,}, valid {:,}, windows {:,}, classes {}".format(n, int(valid.sum()), int((wcount >= 2).sum()),
                                                                    classes), flush=True)

    rng = np.random.default_rng(42)
    windows = np.unique(win[valid])
    nw = len(windows)
    split = {}
    # chronological: 70/10/20 of the windows in time order
    pos = np.searchsorted(windows, win)
    s = np.full(n, -1, dtype="int8")
    s[valid] = np.where(pos[valid] < int(0.7 * nw), 0, np.where(pos[valid] < int(0.7 * nw) + int(0.1 * nw), 1, 2))
    split["chronological"] = s
    # random flow split, stratified by class
    s = np.full(n, -1, dtype="int8")
    for c in range(n_classes):
        idx = np.flatnonzero(valid & (y == c))
        idx = idx[rng.permutation(len(idx))]
        a, b = int(round(0.7 * len(idx))), int(round(0.8 * len(idx)))
        s[idx[:a]], s[idx[a:b]], s[idx[b:]] = 0, 1, 2
    split["random_flow"] = s
    # whole windows at random
    perm = rng.permutation(nw)
    wsplit = np.empty(nw, dtype="int8")
    wsplit[perm[:int(0.7 * nw)]] = 0
    wsplit[perm[int(0.7 * nw):int(0.8 * nw)]] = 1
    wsplit[perm[int(0.8 * nw):]] = 2
    s = np.full(n, -1, dtype="int8")
    s[valid] = wsplit[pos[valid]]
    split["random_window"] = s
    # blocked chronological split: ten consecutive blocks of windows, each split 70/10/20 in time order, for data whose
    # attack types were recorded one after another (the fallback of Section 5.3)
    per = nw // 10
    blk = np.minimum(pos // per, 9)
    lo_b = blk * per
    size = np.where(blk == 9, nw, (blk + 1) * per) - lo_b
    k = pos - lo_b
    s = np.full(n, -1, dtype="int8")
    s[valid] = np.where(k[valid] < (size[valid] * 0.7).astype(int), 0,
                        np.where(k[valid] < (size[valid] * 0.7).astype(int) + (size[valid] * 0.1).astype(int), 1, 2))
    split["blocked"] = s
    SP = tuple(args.splits)

    R = {"dataset": os.path.basename(csv), "n_flows": int(n), "n_valid": int(valid.sum()), "n_windows": int(nw),
         "classes": classes, "window_seconds": args.window_seconds,
         "class_counts": {sp: {c: int(((split[sp] == k) & (y == i)).sum()) for i, c in enumerate(classes)}
                          for sp in SP for k in (2,)},
         "attack_rate": float(lab[valid].mean())}
    chrono = split[args.base]
    tr_c, te_c = chrono == 0, chrono == 2
    R["period_ms"] = [float(start[valid].min()), float(start[valid].max())]
    R["test_start_ms"] = float(start[te_c].min())

    # ------------------------------------------------------------ identity audit (chronological split)
    hosts = np.unique(np.concatenate([src[valid], dst[valid]]))
    attack_src = np.unique(src[valid & (lab == 1)])
    benign_src = np.unique(src[valid & (lab == 0)])
    attack_dst = np.unique(dst[valid & (lab == 1)])
    audit = {"n_addresses": int(len(hosts)), "n_attack_sources": int(len(attack_src)),
             "n_attack_sources_also_benign": int(len(np.intersect1d(attack_src, benign_src))),
             "n_attack_targets": int(len(attack_dst))}
    a_src_share = pd.Series(src[valid & (lab == 1)]).value_counts(normalize=True)
    audit["top_attack_sources_share"] = {str(ip_names[k]): float(v) for k, v in a_src_share.head(10).items()}
    audit["n_attack_sources_for_90pct"] = int(np.searchsorted(np.cumsum(a_src_share.to_numpy()), 0.9) + 1)
    te_att = te_c & (lab == 1)
    train_attackers = np.unique(src[tr_c & (lab == 1)])
    audit["test_attack_from_training_attackers"] = (float(np.isin(src[te_att], train_attackers).mean())
                                                    if te_att.any() else None)
    test_hosts = np.union1d(np.unique(src[te_c]), np.unique(dst[te_c]))
    train_hosts = np.union1d(np.unique(src[tr_c]), np.unique(dst[tr_c]))
    audit["test_hosts"] = int(len(test_hosts))
    audit["test_hosts_seen_in_training"] = int(len(np.intersect1d(test_hosts, train_hosts)))
    te_sources = np.unique(src[te_att])
    audit["test_attack_sources"] = int(len(te_sources))
    audit["test_attack_sources_new"] = int(len(np.setdiff1d(te_sources, train_attackers)))
    audit["test_attack_flows_from_new_sources"] = (float((~np.isin(src[te_att], train_attackers)).mean())
                                                   if te_att.any() else None)
    def key_of(kind, mask):
        if kind == "source address":
            return pd.Series(src[mask])
        if kind == "destination address":
            return pd.Series(dst[mask])
        return pd.Series(src[mask].astype("int64") * len(ip_names) + dst[mask])

    for key_name in ("source address", "destination address", "address pair"):
        tab = pd.crosstab(key_of(key_name, tr_c), y[tr_c])
        major = tab.idxmax(axis=1)
        mapped = key_of(key_name, te_c).map(major)
        pred = mapped.fillna(0).astype(int).to_numpy()
        yt = y[te_c].astype(int)
        audit["probe_" + key_name] = {"macro_f1": macro_f1(yt, pred, n_classes), "binary_f1": binary_f1(yt, pred),
                                      "fpr": float(((pred != 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1))}
        # the same lookup on the test flows whose key was seen in training, where it does not fall back to benign
        seen = mapped.notna().to_numpy()
        ys, ps = yt[seen], pred[seen]
        audit["probe_" + key_name + " (seen)"] = {
            "n_flows": int(seen.sum()), "share_of_test": float(seen.mean()),
            "share_of_test_attacks": float(seen[yt != 0].mean()) if (yt != 0).any() else None,
            "macro_f1": macro_f1(ys, ps, n_classes), "binary_f1": binary_f1(ys, ps),
            "fpr": float(((ps != 0) & (ys == 0)).sum() / max((ys == 0).sum(), 1))}
    R["audit"] = audit
    R["audit_split"] = args.base
    print(json.dumps(audit, indent=1)[:1500], flush=True)
    if args.audit_only:
        # class counts of every requested split, and of every calendar day (UTC), to show when each attack type occurs
        R["class_counts_all"] = {sp: {part: {c: int(((split[sp] == k_) & (y == i)).sum()) for i, c in enumerate(classes)}
                                      for k_, part in ((0, "train"), (1, "val"), (2, "test"))} for sp in SP}
        day = pd.to_datetime(start[valid], unit="ms").strftime("%Y-%m-%d")
        tab = pd.crosstab(day, y[valid])
        R["class_counts_by_day"] = {d: {classes[int(c)]: int(tab.loc[d, c]) for c in tab.columns if tab.loc[d, c] > 0}
                                    for d in tab.index}
        json.dump(R, open(out_path, "w", encoding="utf-8"), indent=1)
        print("wrote", out_path)
        return

    # ------------------------------------------------------------ training samples per split and seed (pass 2)
    def sample_ids(sp, seed):
        r = np.random.default_rng(seed)
        tr = np.flatnonzero(split[sp] == 0)
        ytr = y[tr]
        att_idx = tr[ytr != 0]
        if len(att_idx) <= args.subsample // 2:
            ben = tr[ytr == 0]
            keep_b = r.choice(ben, min(len(ben), args.subsample - len(att_idx)), replace=False)
            return np.sort(np.concatenate([att_idx, keep_b]))
        # attack-heavy data: cap each class, rare classes whole, total about the budget
        counts = np.bincount(ytr, minlength=n_classes)
        cap, budget = 0, args.subsample
        caps = np.sort(counts[counts > 0])
        rem, left = budget, int((counts > 0).sum())
        for c_ in caps:
            if c_ * left <= rem:
                rem -= c_
                left -= 1
            else:
                cap = rem // left
                break
        else:
            cap = int(caps.max())
        out = []
        for c in range(n_classes):
            idx = tr[ytr == c]
            out.append(idx if len(idx) <= cap else r.choice(idx, cap, replace=False))
        return np.sort(np.concatenate(out))

    need = {(sp, sd): sample_ids(sp, sd) for sp in SP for sd in args.seeds}
    all_needed = np.unique(np.concatenate(list(need.values())))
    n_feat = len(feature_cols) + (3 if proto_col is not None else 0)
    X_all = np.empty((len(all_needed), n_feat), dtype="float32")
    offset = 0
    for chunk in read_chunks(csv):
        m = len(chunk)
        lo, hi = np.searchsorted(all_needed, [offset, offset + m])
        if hi > lo:
            X_all[lo:hi] = chunk_features(chunk.iloc[all_needed[lo:hi] - offset], feature_cols, proto_col)
        offset += m
    print("training rows loaded:", len(all_needed), "after {:.0f}s".format(time.perf_counter() - t0), flush=True)

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    def train(sp, sd, name, X, Y):
        if name == "random_forest":
            mdl = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                                         random_state=sd)
            mdl.fit(X, Y)
        else:
            present = np.unique(Y)
            remap = {c: i for i, c in enumerate(present)}
            w = len(Y) / (len(present) * np.bincount(Y, minlength=n_classes)[Y])
            mdl = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.2, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=sd)
            mdl.fit(X, np.array([remap[v] for v in Y]), sample_weight=w)
            mdl.present_ = present
        return mdl

    # ------------------------------------------------------------ train and predict one split at a time (pass 3 per
    # split), so only the ten models of one split are held in memory
    test_ids = {sp: np.flatnonzero(split[sp] == 2) for sp in SP}
    preds = {}
    for sp in SP:
        models = {}
        for sd in args.seeds:
            ids = need[(sp, sd)]
            X = X_all[np.searchsorted(all_needed, ids)]
            Y = y[ids].astype(int)
            for name in args.models:
                models[(sp, sd, name)] = train(sp, sd, name, X, Y)
            del X
            print("trained", sp, sd, "after {:.0f}s".format(time.perf_counter() - t0), flush=True)
        for k in models:
            preds[k] = np.zeros(len(test_ids[sp]), dtype="int16")
        ti = test_ids[sp]
        offset, cursor = 0, 0
        for chunk in read_chunks(csv):
            m = len(chunk)
            hi = np.searchsorted(ti, offset + m)
            if hi > cursor:
                xs = chunk_features(chunk.iloc[ti[cursor:hi] - offset], feature_cols, proto_col)
                for (sp2, sd, name), mdl in models.items():
                    p = mdl.predict(xs)
                    if name == "xgboost":
                        p = mdl.present_[p]
                    preds[(sp2, sd, name)][cursor:hi] = p
                cursor = hi
            offset += m
        del models
        print("predicted", sp, "after {:.0f}s".format(time.perf_counter() - t0), flush=True)
    del X_all

    # ------------------------------------------------------------ scores, own test sets and shared flows
    # classes with training flows in every split: a class absent from one training set (mitm under the blocked split)
    # scores zero there by construction, so the gains are also reported without it
    in_train = {sp: np.bincount(y[split[sp] == 0], minlength=n_classes) > 0 for sp in SP}
    keep_all = np.logical_and.reduce([in_train[sp] for sp in SP])
    R["classes_in_every_training_set"] = [c for c, k in zip(classes, keep_all) if k]
    res = {}
    for (sp, sd, name), p in preds.items():
        yt = y[test_ids[sp]].astype(int)
        r_ = res.setdefault(name, {}).setdefault(sp, {"macro_f1": [], "attack_macro_f1": [], "binary_f1": [],
                                                      "macro_f1_trained_classes": []})
        keep_att = np.ones(n_classes, bool)
        keep_att[0] = False
        r_["macro_f1"].append(macro_f1(yt, p, n_classes))
        r_["attack_macro_f1"].append(macro_f1(yt, p, n_classes, keep=keep_att))
        r_["binary_f1"].append(binary_f1(yt, p))
        r_["macro_f1_trained_classes"].append(macro_f1(yt, p, n_classes, keep=keep_all))
    R["scores"] = res
    shared = {}
    for name in args.models:
        for sp in ("random_flow", "random_window"):
            if sp not in SP or args.base not in SP:
                continue
            common = np.intersect1d(test_ids[sp], test_ids[args.base])
            if len(common) == 0:
                continue
            ia = np.searchsorted(test_ids[sp], common)
            ib = np.searchsorted(test_ids[args.base], common)
            yt = y[common].astype(int)
            wc = win[common]
            ca = [window_counts(yt, preds[(sp, sd, name)][ia].astype(int), wc, n_classes) for sd in args.seeds]
            cb = [window_counts(yt, preds[(args.base, sd, name)][ib].astype(int), wc, n_classes)
                  for sd in args.seeds]
            d, lo, hi = paired_gain(ca, cb, n_boot=args.n_boot)
            d2, lo2, hi2 = paired_gain(ca, cb, n_boot=args.n_boot, keep=keep_all)

            def per_class(counts):
                tp, pc, sc = (sum(c[i].sum(0) for c in counts) for i in range(3))
                f1 = np.where(pc + sc > 0, 2 * tp / np.maximum(pc + sc, 1e-12), 0.0)
                return {c: float(v) for c, v in zip(classes, f1)}

            shared.setdefault(name, {})[sp] = {
                "delta": d, "ci": [lo, hi], "n_flows": int(len(common)),
                "n_windows": int(len(np.unique(wc))), "n_attack": int((yt != 0).sum()),
                "delta_trained_classes": d2, "ci_trained_classes": [lo2, hi2],
                "class_counts": {c: int((yt == i).sum()) for i, c in enumerate(classes)},
                "macro_f1": {sp: float(np.mean([macro_from_counts(*c) for c in ca])),
                             args.base: float(np.mean([macro_from_counts(*c) for c in cb]))},
                "macro_f1_trained_classes": {sp: float(np.mean([macro_from_counts(*c, keep=keep_all) for c in ca])),
                                             args.base: float(np.mean([macro_from_counts(*c, keep=keep_all)
                                                                       for c in cb]))},
                "class_f1": {sp: per_class(ca), args.base: per_class(cb)}}
    R["shared"] = shared
    R["base"] = args.base
    R["splits"] = list(SP)
    R["class_counts_all"] = {sp: {part: {c: int(((split[sp] == k_) & (y == i)).sum()) for i, c in enumerate(classes)}
                                  for k_, part in ((0, "train"), (1, "val"), (2, "test"))} for sp in SP}
    R["seconds"] = time.perf_counter() - t0
    json.dump(R, open(out_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: R[k] for k in ("scores", "shared")}, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
