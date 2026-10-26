"""Robustness analyses. Reads the outputs of the planned runs and of
`run_all.py --extended`, and writes every number the paper quotes from them to
paper/generated/robustness_values.json (plus a readable summary on stdout).

    python scripts/robustness_analysis.py --root D:/CAIDS_work

What it computes
  1. the main models over 20 seeds (Tables 9 and 11), a two-level bootstrap
     (seeds and test windows resampled together) for the paired comparisons,
     a test of the gate's smaller seed SD, and the seeds that label the
     victims' replies as attacks, in the main models and in the variant runs
  2. the forest's training recipe for the neural models, a prior correction and
     per-class offsets tuned on validation, and how much both depend on the
     validation sample (validation windows resampled)
  3. how many flows end after their window has closed
  4. the context-group ablation under both splits, the split protocols on the
     flows their test sets share, and the difference between the gains of the
     context-aware and of the other neural models
  5. window length with mini-batches equalised in flows and aligned periods
  6. probes of possible machine fingerprints, and the session split with
     flow-level and window-level validation, without the age feature or with
     it clipped at its training maximum
  7. smaller checks (K = 1, max aggregation, direction-aware messages,
     H2 with DoS, the share of the split gap that survives on shared flows)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analyze import window_counts, signed, interval  # noqa: E402

PLANNED_SEEDS = [1, 2, 3, 4, 5]
EXTRA_SEEDS = list(range(6, 21))
ALL_SEEDS = PLANNED_SEEDS + EXTRA_SEEDS
LOW_VOLUME = ["Analysis", "Backdoor", "Shellcode", "Worms"]
FINGERPRINT_FIELDS = ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_QUERY_ID"]
NEURAL = [("flow_mlp", "mlp", "Flow-only MLP (B2)"), ("egraphsage", "b3", "E-GraphSAGE-style (B3)"),
          ("centrality_egraphsage", "b4", "Centrality-init. (B4)"), ("cagids_concat", "concat", "CAG-IDS, concat."),
          ("cagids", "cagids", "CAG-IDS, gated")]
CONTEXT_MODELS = ("concat", "cagids")
ATTACKERS = {"175.45.176.0", "175.45.176.1", "175.45.176.2", "175.45.176.3"}
SEED_FIGURE_MODELS = [("mlp", "B2\nflow-only"), ("b3", "B3\nE-GraphSAGE"), ("b4", "B4\ncentrality"),
                      ("concat", "CAG-IDS\nconcat."), ("cagids", "CAG-IDS\ngated"), ("gatv2", "CAG-IDS\nGATv2"),
                      ("c1gate", "C1 + gate\ncontrol")]


def make_seed_figure(V, root):
    """Strip plot of the test macro-F1 of every seed of the seven neural models
    of Table 10, with the seeds that flag the victims' replies marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                         "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    # round 10: every run is coloured by the share of the victims' replies it labels as attacks, so the figure
    # shows the failure as a continuum; crosses still mark the runs above the 50% rule
    cmap = plt.get_cmap("plasma_r")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    for x, (k, _) in enumerate(SEED_FIGURE_MODELS):
        f1s = np.asarray(V["_f1s_" + k])
        flag = V["_reply"][k]
        share = np.asarray(V["_reply_shares"][k])
        order = np.argsort(np.argsort(f1s))
        jitter = (order / max(len(f1s) - 1, 1) - 0.5) * 0.36
        sc = ax.scatter(x + jitter[~flag], f1s[~flag], s=12, c=share[~flag], cmap=cmap, norm=norm, linewidths=0.3,
                        edgecolors="#555555", zorder=3, label="run" if x == 0 else None)
        if flag.any():
            ax.scatter(x + jitter[flag], f1s[flag], s=18, marker="x", c=share[flag], cmap=cmap, norm=norm,
                       linewidths=1.1, zorder=4, label="run above the 50% rule" if k == "b3" else None)
        ax.hlines(np.median(f1s), x - 0.27, x + 0.27, color="black", lw=1.1, zorder=5,
                  label="median" if x == 0 else None)
        ax.scatter([x + 0.31], [f1s.mean()], marker="D", s=10, facecolor="white", edgecolor="black", linewidths=0.7,
                   zorder=5, label="mean" if x == 0 else None)
    ax.set_xticks(range(len(SEED_FIGURE_MODELS)))
    ax.set_xticklabels([lab for _, lab in SEED_FIGURE_MODELS])
    ax.set_ylabel("test macro-F1 (chronological split)")
    ax.set_xlim(-0.6, len(SEED_FIGURE_MODELS) - 0.4)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
                    columnspacing=1.0)
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        if t.get_text() == "run":
            h.set_facecolor("#c8c8c8")
        elif t.get_text() == "median":
            h.set_color("black")
        elif "50%" in t.get_text():
            h.set_color("#2c0d6b")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("share of victims' replies\nlabelled as attacks", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    out = Path(root) / "paper" / "generated" / "figures" / "fig_seeds.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("figure written to", out)


def f3(x):
    return "{:.3f}".format(x)


def pm(values):
    values = np.asarray(values, dtype=float)
    return "{:.3f} ± {:.3f}".format(values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0)


def ci_text(res, key="all"):
    r = res[key]
    return "{} {}".format(signed(r["delta"]), interval(*r["two_level"]))


def excludes_zero(lo_hi):
    return "yes" if (lo_hi[0] > 0 or lo_hi[1] < 0) else "no"


def load_json(results, name):
    path = results / (name + ".json")
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def runs_of(results, name):
    payload = load_json(results, name)
    return payload["runs"] if payload else None


def runs_all(results, tag):
    """Runs of gnn_<tag> (seeds 1 to 5) followed by those of gnn_extra_<tag>
    (seeds 6 to 20) when that file exists."""
    base = runs_of(results, "gnn_" + tag)
    if base is None:
        return None
    return base + (runs_of(results, "gnn_extra_" + tag) or [])


def macro_list(runs, key="macro_f1"):
    return [r[key] for r in runs]


def load_pred(path):
    """(ids, y_true, y_pred) sorted by flow id, in compact types: many models
    with 20 seeds each are held in memory at once."""
    data = np.load(path)
    order = np.argsort(data["edge_ids"])
    return (data["edge_ids"][order].astype("int32"), data["y_true"][order].astype("int8"),
            data["y_pred"][order].astype("int8"))


def pred_paths(pred_dir, tag, seeds):
    return [pred_dir / "{}_seed{}.npz".format(tag, s) for s in seeds]


def seed_preds(pred_dir, tag, extra_tag=None, seeds_planned=PLANNED_SEEDS, seeds_extra=EXTRA_SEEDS):
    """Predictions of the planned seeds, followed by those of the extra seeds
    when they all exist. Returns a list of (ids, y_true, y_pred)."""
    files = pred_paths(pred_dir, tag, seeds_planned)
    if extra_tag:
        extra = pred_paths(pred_dir, extra_tag, seeds_extra)
        if all(f.exists() for f in extra):
            files += extra
    if not all(f.exists() for f in files):
        return None
    out = [load_pred(f) for f in files]
    base = out[0][0]  # every seed of a model is scored on the same flows: keep one id array
    return [(base if np.array_equal(i, base) else i, y, p) for i, y, p in out]


def preds_all(pred_dir, tag):
    return seed_preds(pred_dir, tag, "extra_" + tag)


# ----------------------------------------------------------------------------
# 1. two-level bootstrap of any linear combination of seed-averaged macro-F1


def block_samples(n, n_boot, block, rng):
    starts_all = np.arange(max(n - block + 1, 1))
    n_blocks = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out.append(idx[idx < n])
    return out


def stacked_counts(preds, edge_window, windows, n_classes):
    tps, fps, fns = [], [], []
    for ids, y, p in preds:
        tp, fp, fn = window_counts(ids, y, p, edge_window, windows, n_classes)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    return np.stack(tps), np.stack(fps), np.stack(fns)  # (seeds, windows, classes)


def macro_per_seed(tp, fp, fn, idx, attacks=False):
    """Macro-F1 of every seed on the windows idx (with repetition)."""
    t, f, n = tp[:, idx, :].sum(1), fp[:, idx, :].sum(1), fn[:, idx, :].sum(1)
    if attacks:
        t, f, n = t[:, 1:], f[:, 1:], n[:, 1:]
    support = (t + n)[0] > 0
    denom = 2 * t + f + n
    f1 = np.where(denom > 0, 2 * t / np.maximum(denom, 1e-12), 0.0)
    return f1[:, support].mean(axis=1)


def restrict(preds, keep_ids):
    if keep_ids is None:
        return preds
    out = []
    for ids, y, p in preds:
        m = np.isin(ids, keep_ids, assume_unique=True)
        out.append((ids[m], y[m], p[m]))
    return out


SEED_STATS = {"mean": np.mean, "median": np.median, "trim20": lambda x: stats.trim_mean(x, 0.2)}
BONFERRONI_3 = [100 * 0.025 / 3, 100 * (1 - 0.025 / 3)]   # 95% family-wise over the three planned comparisons
# round 6: smallest effect of interest in macro-F1, fixed after the results (so the equivalence tests are
# exploratory), and the z-multiplier of a minimum detectable effect (two-sided 5% level, 80% power)
SESOI = 0.02
MDE_Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)


def contrast(groups, coefs, edge_window, n_classes, keep_ids=None, n_boot=2000, block=10, seed=0, extra_stats=()):
    """groups: one list of seed predictions per model, all scored on the same
    flows (after restriction to keep_ids). The statistic is sum_g coef_g times
    the seed-averaged macro-F1 of group g. Each replicate resamples blocks of
    test windows (shared by all groups) and, independently for each group, its
    seeds with replacement. With coefs [1, -1] this is the paired two-level
    bootstrap of a difference; with four or more groups it gives a difference
    of differences. extra_stats ("median", "trim20") repeat the two-level
    interval with the seed mean replaced by a robust statistic over seeds, on
    the same replicates."""
    groups = [restrict(g, keep_ids) for g in groups]
    ids0 = groups[0][0][0]
    for g in groups:
        for x in g:
            if not np.array_equal(x[0], ids0):
                raise ValueError("models scored on different flows")
    windows = np.unique(edge_window[ids0])
    counts = [stacked_counts(g, edge_window, windows, n_classes) for g in groups]
    n = len(windows)
    rng = np.random.default_rng(seed)
    samples = block_samples(n, n_boot, block, rng)
    full = np.arange(n)
    out = {"n_seeds": [len(g) for g in groups], "n_windows": int(n), "n_flows": int(len(ids0))}
    for attacks, key in ((False, "all"), (True, "attacks")):
        f_full = [macro_per_seed(*c, full, attacks) for c in counts]
        point = float(sum(w * f.mean() for w, f in zip(coefs, f_full)))
        two, win_only, seed_only = [], [], []
        robust = {s: [] for s in extra_stats}
        for idx in samples:
            fs = [macro_per_seed(*c, idx, attacks) for c in counts]
            picks = [rng.integers(0, len(f), len(f)) for f in fs]
            two.append(sum(w * f[r].mean() for w, f, r in zip(coefs, fs, picks)))
            win_only.append(sum(w * f.mean() for w, f in zip(coefs, fs)))
            seed_only.append(sum(w * f[r].mean() for w, f, r in zip(coefs, f_full, picks)))
            for s in extra_stats:
                robust[s].append(sum(w * SEED_STATS[s](f[r]) for w, f, r in zip(coefs, fs, picks)))
        two, win_only, seed_only = map(np.asarray, (two, win_only, seed_only))
        out[key] = {
            "delta": point,
            "means": [float(f.mean()) for f in f_full],
            "sds": [float(f.std(ddof=1)) if len(f) > 1 else 0.0 for f in f_full],
            "per_seed": [f.tolist() for f in f_full],
            "two_level": [float(v) for v in np.percentile(two, [2.5, 97.5])],
            "two_level_bonf3": [float(v) for v in np.percentile(two, BONFERRONI_3)],
            # round 6: the spread of the two-level replicates (for a minimum detectable effect) and the 90%
            # interval of the two one-sided tests of equivalence
            "two_level_sd": float(two.std(ddof=1)),
            "two_level_90": [float(v) for v in np.percentile(two, [5, 95])],
            "window_only": [float(v) for v in np.percentile(win_only, [2.5, 97.5])],
            "seed_only": [float(v) for v in np.percentile(seed_only, [2.5, 97.5])],
            "p_two_level_le_0": float((two <= 0).mean()),
            "robust": {s: {"delta": float(sum(w * SEED_STATS[s](f) for w, f in zip(coefs, f_full))),
                           "two_level": [float(v) for v in np.percentile(robust[s], [2.5, 97.5])]}
                       for s in extra_stats},
        }
    return out


def two_level_bootstrap(preds_a, preds_b, edge_window, n_classes, n_boot=2000, block=10, seed=0, keep_ids=None,
                        extra_stats=()):
    res = contrast([preds_a, preds_b], [1, -1], edge_window, n_classes, keep_ids, n_boot, block, seed, extra_stats)
    for key in ("all", "attacks"):
        r = res[key]
        r["a_mean"], r["b_mean"] = r["means"]
        r["a_sd"], r["b_sd"] = r["sds"]
        fa, fb = r["per_seed"]
        r["welch_p"] = float(stats.ttest_ind(fa, fb, equal_var=False).pvalue) if min(len(fa), len(fb)) > 1 else 1.0
    res["n_seeds_a"], res["n_seeds_b"] = res["n_seeds"]
    return res


def variance_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ratio = a.var(ddof=1) / b.var(ddof=1)
    dfa, dfb = len(a) - 1, len(b) - 1
    p_f = 2 * min(stats.f.cdf(ratio, dfa, dfb), 1 - stats.f.cdf(ratio, dfa, dfb))
    crit = stats.f.ppf(0.975, dfa, dfb)
    bf = stats.levene(a, b, center="median")
    return {"sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)), "var_ratio": float(ratio),
            "f_crit_975": float(crit), "p_f": float(p_f), "p_brown_forsythe": float(bf.pvalue)}


def welch_interval(a, b, level=0.95):
    """Difference of means a - b with a Welch t interval (seed noise only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    d = a.mean() - b.mean()
    return float(d), float(d - t * se), float(d + t * se)


def hodges_lehmann(a, b, level=0.95):
    """Rank-based shift a - b over seeds: the Hodges-Lehmann estimate (median of
    all pairwise differences), its distribution-free interval from the
    Mann-Whitney critical value (normal approximation), the two-sided
    Mann-Whitney p-value and the probability that a seed of a beats one of b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.sort((a[:, None] - b[None, :]).ravel())
    n, m = len(a), len(b)
    z = stats.norm.ppf(0.5 + level / 2)
    k = int(np.floor(n * m / 2.0 - z * np.sqrt(n * m * (n + m + 1) / 12.0)))
    k = min(max(k, 0), len(d) - 1)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"hl": float(np.median(d)), "lo": float(d[k]), "hi": float(d[len(d) - 1 - k]), "p": float(mw.pvalue),
            "p_superior": float(mw.statistic / (n * m))}


# ----------------------------------------------------------------------------
# 2. prior correction and per-class offsets tuned on validation


def macro_f1_from_pred(y, p, n_classes, w=None):
    present = np.bincount(y, weights=w, minlength=n_classes) > 0
    hit = y == p
    tp = np.bincount(y[hit], weights=None if w is None else w[hit], minlength=n_classes).astype(float)
    pc = np.bincount(p, weights=w, minlength=n_classes).astype(float)
    sc = np.bincount(y, weights=w, minlength=n_classes).astype(float)
    denom = pc + sc
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return float(f1[present].mean())


LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)
GRID = np.linspace(-6, 6, 25)


def prior_correction(val_scores, val_y, test_scores, test_y, log_w, n_classes, w=None):
    """Scores are logits (or log-probabilities). The correction subtracts
    lambda * log(class weight) from each class score, which undoes the weighting
    for lambda = 1. Lambda is chosen on validation macro-F1 (optionally with
    flow weights, for a resampled validation set)."""
    best, best_lam = -1.0, 0.0
    curve = []
    for lam in LAMBDAS:
        s = macro_f1_from_pred(val_y, (val_scores - lam * log_w).argmax(1), n_classes, w)
        curve.append(s)
        if s > best + 1e-9:
            best, best_lam = s, float(lam)
    test_plain = macro_f1_from_pred(test_y, test_scores.argmax(1), n_classes)
    test_full = macro_f1_from_pred(test_y, (test_scores - log_w).argmax(1), n_classes)
    pred_tuned = (test_scores - best_lam * log_w).argmax(1)
    test_tuned = macro_f1_from_pred(test_y, pred_tuned, n_classes)
    return {"lambda": best_lam, "test_plain": test_plain, "test_full": test_full, "test_tuned": test_tuned,
            "val_curve": curve, "pred_tuned": pred_tuned}


def _macro_from_rows(tp, pc, sc, present):
    denom = pc + sc
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return f1[..., present].mean(axis=-1)


def fit_offsets(val_scores, val_y, n_classes, w=None, grid=GRID, passes=2):
    """Per-class additive offsets to the scores, tuned on validation by
    coordinate ascent (a threshold correction); benign stays at zero. Flow
    weights allow a resampled validation set. A flow whose two largest scores
    differ by more than the widest possible offset difference can never change
    class, so it enters as a constant count; for every class, all grid values
    are scored at once, and the selection follows the sequential search of the
    original procedure, including argmax's tie-breaking by class index."""
    w = np.ones(len(val_y)) if w is None else np.asarray(w, float)
    sc = np.bincount(val_y, weights=w, minlength=n_classes)
    present = sc > 0
    top2 = np.partition(val_scores, -2, axis=1)[:, -2:]
    mobile = (top2[:, 1] - top2[:, 0]) <= (grid.max() - grid.min()) + 1e-9
    fp_ = val_scores[~mobile].argmax(1)
    fy, fw = val_y[~mobile], w[~mobile]
    hit = fp_ == fy
    tp_fix = np.bincount(fy[hit], weights=fw[hit], minlength=n_classes)
    pc_fix = np.bincount(fp_, weights=fw, minlength=n_classes)
    S, y, wm = val_scores[mobile], val_y[mobile], w[mobile]
    eye = np.eye(n_classes)
    bias = np.zeros(n_classes)

    def score(b):
        p = (S + b).argmax(1)
        h = p == y
        tp = tp_fix + np.bincount(y[h], weights=wm[h], minlength=n_classes)
        pc = pc_fix + np.bincount(p, weights=wm, minlength=n_classes)
        return float(_macro_from_rows(tp, pc, sc, present))

    best = score(bias)
    rows = np.arange(len(S))
    for _ in range(passes):
        for c in range(1, n_classes):
            other = S + bias
            other[:, c] = -np.inf
            alt = other.argmax(1)
            m = other[rows, alt]
            cand = S[:, c][:, None] + grid[None, :]
            P = (cand > m[:, None]) | ((cand == m[:, None]) & (c < alt)[:, None])
            Pw = P * wm[:, None]
            notP = (~P) * wm[:, None]
            A = eye[alt]
            pc_g = pc_fix[None, :] + notP.T @ A
            tp_g = tp_fix[None, :] + notP.T @ (A * (y == alt)[:, None])
            pc_g[:, c] += Pw.sum(0)
            tp_g[:, c] += (Pw * (y == c)[:, None]).sum(0)
            scores = _macro_from_rows(tp_g, pc_g, sc, present)
            keep = bias[c]
            for g, s in zip(grid, scores):
                if s > best + 1e-9:
                    best, keep = float(s), float(g)
            bias[c] = keep
    return bias, best


def per_class_pr(y, p, n_classes):
    out = []
    for c in range(n_classes):
        tp = np.sum((y == c) & (p == c))
        pp = np.sum(p == c)
        sp = np.sum(y == c)
        out.append((tp / pp if pp else 0.0, tp / sp if sp else 0.0))
    return out


def load_correction_inputs(path, kind):
    d = np.load(path)
    if kind == "neural":
        if "val_y_logit" not in d.files:
            return None
        order_v, order_t = np.argsort(d["val_edge_ids"]), np.argsort(d["edge_ids"])
        val_ids = d["val_edge_ids"][order_v]
        val_scores = d["val_y_logit"].astype("float64")[order_v]
        val_y = d["val_y_true"].astype(int)[order_v]
        test_scores = d["y_logit"].astype("float64")[order_t]
        test_y = d["y_true"].astype(int)[order_t]
        log_w = np.log(d["class_weights"].astype("float64"))
    else:
        if "val_y_prob" not in d.files:
            return None
        order_v, order_t = np.argsort(d["val_edge_ids"]), np.argsort(d["edge_ids"])
        val_ids = d["val_edge_ids"][order_v]
        val_scores = np.log(d["val_y_prob"].astype("float64")[order_v] + 1e-6)
        val_y = d["val_y_true"].astype(int)[order_v]
        test_scores = np.log(d["test_y_prob32"].astype("float64")[order_t] + 1e-6)
        test_y = d["y_true"].astype(int)[order_t]
        log_w = None
    return val_ids, val_scores, val_y, test_scores, test_y, log_w


def offset_scorer(test_scores, test_y, n_classes, span=GRID.max() - GRID.min()):
    """Test macro-F1 for a vector of class offsets. Test flows that no offset
    on the grid can move to another class enter as constant counts."""
    top2 = np.partition(test_scores, -2, axis=1)[:, -2:]
    mobile = (top2[:, 1] - top2[:, 0]) <= span + 1e-9
    fp_, fy = test_scores[~mobile].argmax(1), test_y[~mobile]
    hit = fp_ == fy
    tp_fix = np.bincount(fy[hit], minlength=n_classes).astype(float)
    pc_fix = np.bincount(fp_, minlength=n_classes).astype(float)
    sc = np.bincount(test_y, minlength=n_classes).astype(float)
    S, y = test_scores[mobile], test_y[mobile]

    def score(bias):
        p = (S + bias).argmax(1)
        h = p == y
        tp = tp_fix + np.bincount(y[h], minlength=n_classes)
        pc = pc_fix + np.bincount(p, minlength=n_classes)
        return float(_macro_from_rows(tp, pc, sc, sc > 0))
    return score


def correction_task(job):
    """Point estimates of both corrections for one trained model and, if asked,
    refits of the per-class offsets on resampled validation sets."""
    path, kind, forest_log_w, val_window_of, val_samples, n_classes, do_boot = job
    loaded = load_correction_inputs(path, kind)
    if loaded is None:
        return None
    val_ids, vs, vy, ts, ty, log_w = loaded
    if log_w is None:
        log_w = forest_log_w
    res = prior_correction(vs, vy, ts, ty, log_w, n_classes)
    bias, _ = fit_offsets(vs, vy, n_classes)
    out = {"plain": res["test_plain"], "full": res["test_full"], "tuned": res["test_tuned"], "lambda": res["lambda"],
           "worms_pr": per_class_pr(ty, res["pred_tuned"], n_classes)[n_classes - 1],
           "offsets": macro_f1_from_pred(ty, (ts + bias).argmax(1), n_classes), "bias": bias.tolist(),
           "boot_offsets": []}
    if do_boot:
        windows, win_of_flow = val_window_of(val_ids)
        score = offset_scorer(ts, ty, n_classes)
        for idx in val_samples:
            w = np.bincount(idx, minlength=len(windows)).astype(float)[win_of_flow]
            b, _ = fit_offsets(vs, vy, n_classes, w=w)
            out["boot_offsets"].append(score(b))
    return out


class ValWindows:
    """Maps validation flow ids to the index of their window (picklable)."""

    def __init__(self, edge_window, val_ids_all):
        self.windows = np.unique(edge_window[val_ids_all])
        self.ids = np.sort(val_ids_all)
        self.win = np.searchsorted(self.windows, edge_window[self.ids])

    def __call__(self, val_ids):
        pos = np.searchsorted(self.ids, val_ids)
        if not np.array_equal(self.ids[pos], val_ids):
            raise ValueError("validation flows differ between models")
        return self.windows, self.win[pos]


CORRECTION_ROWS = [("mlp_all", "Flow-only MLP (B2), all training flows", "extra_flow_mlp", EXTRA_SEEDS, "neural"),
                   ("mlp_sub", "Flow-only MLP (B2), forest's sample", "sub_flow_mlp", PLANNED_SEEDS, "neural"),
                   ("b4_all", "Centrality-initialised GNN (B4), all training flows", "extra_centrality_egraphsage",
                    EXTRA_SEEDS, "neural"),
                   ("b4_sub", "Centrality-initialised GNN (B4), forest's sample", "sub_centrality_egraphsage",
                    PLANNED_SEEDS, "neural"),
                   ("cagids_all", "CAG-IDS, gated, all training flows", "extra_cagids", EXTRA_SEEDS, "neural"),
                   ("cagids_sub", "CAG-IDS, gated, forest's sample", "sub_cagids", PLANNED_SEEDS, "neural"),
                   ("rf", "Random forest, its own sample", "random_forest_withval", PLANNED_SEEDS, "forest")]


def run_corrections(args, work, results, pred_dir, meta, classes, edge_window):
    """Point estimates and validation-resampled refits of both corrections for
    every model of Table 17, cached next to the results because they take a while."""
    n_classes = len(classes)
    # The forest's balanced-subsample weights are inverse class frequencies in its
    # 400,000-flow training sample (all training attack flows plus benign flows).
    train_counts = np.array([meta["class_counts"]["train"][c] for c in classes], dtype="float64")
    sub_counts = train_counts.copy()
    sub_counts[0] = 400000 - train_counts[1:].sum()
    forest_log_w = np.log(sub_counts.sum() / (n_classes * sub_counts))
    val_ids_all = np.flatnonzero(np.load(work / "graphs" / "edge_split.npy") == 1)
    val_window_of = ValWindows(edge_window, val_ids_all)
    val_samples = block_samples(len(val_window_of.windows), args.n_val_boot, 10, np.random.default_rng(1))
    jobs, owners = [], []
    for key, label, tag, seeds, kind in CORRECTION_ROWS:
        for k, s in enumerate(seeds):
            # the validation resamples are refitted for the first five seeds of each row
            jobs.append((pred_dir / "{}_seed{}.npz".format(tag, s), kind, forest_log_w, val_window_of, val_samples,
                         n_classes, k < 5))
            owners.append(key)
    cache = results / "robustness_corrections_cache.json"
    stamp = [[str(j[0]), os.path.getmtime(j[0]) if os.path.exists(j[0]) else None] for j in jobs] + [args.n_val_boot]
    outs = None
    if cache.exists():
        saved = json.load(open(cache, encoding="utf-8"))
        if saved.get("stamp") == stamp:
            outs = saved["outs"]
    if outs is None:
        if args.workers <= 1:  # in-process: no copies of the job arrays
            outs = []
            for k, job in enumerate(jobs):
                outs.append(correction_task(job))
                print("correction refits {}/{}".format(k + 1, len(jobs)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                outs = list(pool.map(correction_task, jobs))
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump({"stamp": stamp, "outs": outs}, handle, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return CORRECTION_ROWS, owners, outs


# ----------------------------------------------------------------------------


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=os.path.join(ROOT, "paper", "generated", "robustness_values.json"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-val-boot", type=int, default=100, help="validation resamples for the corrections")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--corrections-only", action="store_true", help="only compute and cache the correction refits")
    parser.add_argument("--skip-corrections", action="store_true", help="leave out the correction refits (quick look)")
    args = parser.parse_args()

    root = Path(args.root)
    work = root / "outputs" / "work"
    results = root / "outputs" / "results"
    pred_dir = results / "predictions"
    meta = json.load(open(work / "graphs" / "meta.json", encoding="utf-8"))
    classes = meta["classes"]
    n_classes = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    edge_window = np.load(work / "graphs" / "graphs.npz")["edge_window"]
    V = {}

    def put(key, value):
        V[key] = value
        print("{:<48} {}".format(key, value if not isinstance(value, (dict, list)) else json.dumps(value)[:160]))

    if args.corrections_only:
        run_corrections(args, work, results, pred_dir, meta, classes, edge_window)
        print("corrections cached in", results / "robustness_corrections_cache.json")
        return

    # ------------------------------------------------------------------ 1
    print("\n=== 1. the main models over 20 seeds and a two-level bootstrap ===")
    tags = {"mlp": "flow_mlp", "b3": "egraphsage", "b4": "centrality_egraphsage", "concat": "cagids_concat",
            "cagids": "cagids", "c1gate": "c1_gate", "gatv2": "cagids_gatv2"}
    preds = {k: preds_all(pred_dir, t) for k, t in tags.items()}
    preds["rf"] = seed_preds(pred_dir, "random_forest")
    for k, p in preds.items():
        if p is None:
            print("missing predictions for", k)
            continue
        f1s = [macro_f1_from_pred(y, q, n_classes) for _, y, q in p]
        put("s20_{}_n".format(k), len(p))
        put("s20_{}_macro".format(k), pm(f1s))
        put("s20_{}_macro_mean".format(k), f3(np.mean(f1s)))
        put("s20_{}_macro_sd".format(k), f3(np.std(f1s, ddof=1)))
        put("s20_{}_macro_range".format(k), "{} to {}".format(f3(min(f1s)), f3(max(f1s))))
        put("s5_{}_macro".format(k), pm(f1s[:5]))
        V["_f1s_" + k] = f1s

    # Per-seed view (round 3). Flows sent back to the attackers are labelled benign (Section 6.1); a seed
    # "flags the replies" when it calls more than half of these benign test flows attacks.
    dst_ip = pd.read_parquet(work / "flows.parquet", columns=["dst_ip"])["dst_ip"].to_numpy(object)
    to_attacker = np.isin(dst_ip[np.load(work / "graphs" / "row_index.npy")], sorted(ATTACKERS))
    del dst_ip
    V["_reply"] = {}
    for k, p in preds.items():
        if p is None or k == "rf":
            continue
        shares = []
        for ids, y, q in p:
            m = (y == 0) & to_attacker[ids]
            shares.append(float((q[m] != 0).mean()))
        flag = np.array(shares) > 0.5
        f1s = np.array(V["_f1s_" + k])
        V["_reply"][k] = flag
        V.setdefault("_reply_shares", {})[k] = shares
        # round 10: the share is a continuum, so also count the runs between 10% and 50% and give the spread
        sh = np.array(shares)
        put("reply_{}_runs_10_50".format(k), int(((sh > 0.10) & (sh <= 0.5)).sum()))
        put("reply_{}_share_median".format(k), "{:.0f}%".format(100 * np.median(sh)))
        put("reply_{}_share_below_50_max".format(k), "{:.0f}%".format(100 * sh[sh <= 0.5].max()) if (sh <= 0.5).any()
            else "–")
        put("reply_{}_n_flows".format(k), int(((p[0][1] == 0) & to_attacker[p[0][0]]).sum()))
        put("reply_{}_n".format(k), int(flag.sum()))
        put("reply_{}_share_flagged".format(k), "{:.1f}% to {:.1f}%".format(100 * min(np.array(shares)[flag], default=0),
                                                                            100 * max(np.array(shares)[flag], default=0)))
        put("med_{}".format(k), f3(np.median(f1s)))
        put("iqr_{}".format(k), "{} to {}".format(*[f3(v) for v in np.percentile(f1s, [25, 75])]))
        if flag.any():
            put("reply_{}_macro_range".format(k), "{} to {}".format(f3(f1s[flag].min()), f3(f1s[flag].max())))
            put("reply_{}_mean_without".format(k), f3(f1s[~flag].mean()))
            put("reply_{}_sd_without".format(k), f3(f1s[~flag].std(ddof=1)))
    for k, t in tags.items():
        if k not in V["_reply"]:
            continue
        rr = runs_all(results, t)
        flag = V["_reply"][k]
        if flag.any():
            put("reply_{}_binary".format(k), "{} to {}".format(f3(min(r["binary_f1"] for r, f in zip(rr, flag) if f)),
                                                               f3(max(r["binary_f1"] for r, f in zip(rr, flag) if f))))
            put("reply_{}_fpr".format(k), "{:.2f}% to {:.2f}%".format(
                100 * min(r["false_positive_rate"] for r, f in zip(rr, flag) if f),
                100 * max(r["false_positive_rate"] for r, f in zip(rr, flag) if f)))
            put("reply_{}_best_epoch".format(k), "{} to {}".format(min(r["best_epoch"] for r, f in zip(rr, flag) if f),
                                                                   max(r["best_epoch"] for r, f in zip(rr, flag) if f)))
        others = [r["binary_f1"] for r, f in zip(rr, flag) if not f]
        put("reply_{}_binary_min_others".format(k), f3(min(others)))
    put("reply_total_flagging_seeds", int(sum(V["_reply"][k].sum() for k in V["_reply"])))
    flagged_epochs = [r["best_epoch"] for k in ("b3", "b4") if k in V["_reply"]
                      for r, f in zip(runs_all(results, tags[k]), V["_reply"][k]) if f]
    if flagged_epochs:
        put("reply_best_epoch_median", int(np.median(flagged_epochs)))
    make_seed_figure(V, ROOT)

    # round 4: binomial (Clopper-Pearson) intervals for the failure rates of the main models, the same failure
    # in the variant runs of the 60-s chronological split, and what the planned B3 seeds that failed do when
    # they are trained longer with the same seed
    def cp_interval(k, n, level=0.95):
        a = (1 - level) / 2
        lo = 0.0 if k == 0 else stats.beta.ppf(a, k, n - k + 1)
        hi = 1.0 if k == n else stats.beta.ppf(1 - a, k + 1, n - k)
        return float(lo), float(hi)

    rate_k = {}
    for name, ks in (("b3b4", ("b3", "b4")), ("others", ("mlp", "concat", "cagids", "gatv2", "c1gate"))):
        k_ = int(sum(V["_reply"][k].sum() for k in ks if k in V["_reply"]))
        n_ = int(sum(len(V["_reply"][k]) for k in ks if k in V["_reply"]))
        lo, hi = cp_interval(k_, n_)
        rate_k[name] = (k_, n_)
        put("reply_rate_{}".format(name), "{} of {}".format(k_, n_))
        put("reply_rate_{}_pct".format(name), "{:.1f}%".format(100 * k_ / n_))
        put("reply_rate_{}_ci".format(name), "{:.0f}% to {:.0f}%".format(100 * lo, 100 * hi))
    if len(rate_k) == 2:
        (kb, nb), (ko, no) = rate_k["b3b4"], rate_k["others"]
        put("reply_rate_fisher_p", "{:.1e}".format(stats.fisher_exact([[kb, nb - kb], [ko, no - ko]])[1]))

    def reply_flags(p):
        flags = []
        for ids, y, q in p:
            m = (y == 0) & to_attacker[ids]
            flags.append(float((q[m] != 0).mean()) > 0.5)
        return np.array(flags)

    variants = [("long_b3", "long_egraphsage"), ("long_b4", "long_centrality_egraphsage"), ("long_mlp", "long_flow_mlp"),
                ("long_concat", "long_cagids_concat"), ("long_cagids", "long_cagids"),
                ("sw_b3", "sw_egraphsage"), ("sw_b4", "sw_centrality_egraphsage"), ("sw_mlp", "sw_flow_mlp"),
                ("sw_concat", "sw_cagids_concat"), ("sw_cagids", "sw_cagids"),
                ("focal_b3", "focal_egraphsage"), ("focal_b4", "focal_centrality_egraphsage"),
                ("focal_mlp", "focal_flow_mlp"), ("focal_concat", "focal_cagids_concat"), ("focal_cagids", "cagids_focal"),
                ("k1_b3", "k1_egraphsage"), ("k1_cagids", "k1_cagids"), ("max_b3", "max_egraphsage"),
                ("max_cagids", "max_cagids"), ("dir_b3", "dir_egraphsage"), ("dir_cagids", "dir_cagids"),
                ("sub_mlp", "sub_flow_mlp"), ("sub_b4", "sub_centrality_egraphsage"), ("sub_cagids", "sub_cagids"),
                ("no_c1", "abl_no_structural"), ("no_c2", "abl_no_behavioural"), ("no_c3", "abl_no_role"),
                ("with_c4", "abl_with_time"), ("no_age", "noage_cagids"), ("probe", "context_only")]
    var_flags = {}
    for k, t in variants:
        p = preds_all(pred_dir, t)
        if p is None:
            continue
        var_flags[k] = reply_flags(p)
        put("reply_var_{}".format(k), "{} of {}".format(int(var_flags[k].sum()), len(var_flags[k])))
        del p
    for name, ks in (("long_b3b4", ("long_b3", "long_b4")),):
        if all(k in var_flags for k in ks):
            put("reply_var_{}".format(name), "{} of {}".format(int(sum(var_flags[k].sum() for k in ks)),
                                                               sum(len(var_flags[k]) for k in ks)))
    rr_b3, rr_long = runs_of(results, "gnn_egraphsage"), runs_of(results, "gnn_long_egraphsage")
    failed = [i for i, f in enumerate(V["_reply"].get("b3", [])[:len(PLANNED_SEEDS)]) if f]
    if failed and rr_b3 and rr_long and "long_b3" in var_flags:
        put("reply_b3_planned_failed_seeds", ", ".join(str(rr_b3[i]["seed"]) for i in failed))
        put("reply_b3_planned_failed_best_epoch", ", ".join(str(rr_b3[i]["best_epoch"]) for i in failed))
        put("reply_b3_long_same_seed_best_epoch", ", ".join(str(rr_long[i]["best_epoch"]) for i in failed))
        put("reply_b3_long_same_seed_failed", "yes" if any(var_flags["long_b3"][i] for i in failed) else "no")
    # gated CAG-IDS against B3 in the variants in which no B3 seed fails (five seeds each)
    for name, a, b in (("long", "long_cagids", "long_egraphsage"), ("sw", "sw_cagids", "sw_egraphsage"),
                       ("dir", "dir_cagids", "dir_egraphsage")):
        pa, pb = seed_preds(pred_dir, a), seed_preds(pred_dir, b)
        if pa is None or pb is None:
            continue
        res = two_level_bootstrap(pa, pb, edge_window, n_classes, n_boot=args.n_boot)
        put("tl_{}_cagids_b3".format(name), ci_text(res))
        put("tl_{}_cagids_b3_excludes_zero".format(name), excludes_zero(res["all"]["two_level"]))
        put("tl_{}_cagids_b3_seeds".format(name), "{} vs {}".format(res["n_seeds_a"], res["n_seeds_b"]))
        del pa, pb
    # round 5: the seeds 6 to 20 of B3 and B4 that fail under the planned settings, retrained with the same seed
    # and the longer budget (60 epochs, patience 12). Does the same seed still fail?
    rerun = {}
    for k, tag, planned_tag in (("b3", "failrerun_egraphsage", "egraphsage"),
                                ("b4", "failrerun_centrality_egraphsage", "centrality_egraphsage")):
        rr = runs_of(results, "gnn_" + tag)
        if not rr or k not in V["_reply"]:
            continue
        seeds = [r["seed"] for r in rr]
        planned = {r["seed"]: (r, f) for r, f in zip(runs_all(results, planned_tag), V["_reply"][k])}
        assert all(planned[s][1] for s in seeds), "a rerun seed did not fail under the planned settings"
        assert sorted(seeds) == sorted(s for s, (r, f) in planned.items() if f and s > len(PLANNED_SEEDS))
        shares = []
        for ids, y, q in seed_preds(pred_dir, tag, seeds_planned=seeds):
            m = (y == 0) & to_attacker[ids]
            shares.append(float((q[m] != 0).mean()))
        flags = np.array(shares) > 0.5
        put("rerun_{}_max_share".format(k), "{:.0f}%".format(100 * max(shares)))
        rerun[k] = (flags, rr, [planned[s][0] for s in seeds])
        put("rerun_{}_seeds".format(k), ", ".join(str(s) for s in seeds))
        put("rerun_{}_failed".format(k), "{} of {}".format(int(flags.sum()), len(flags)))
        put("rerun_{}_best_epoch".format(k), "{} to {}".format(min(r["best_epoch"] for r in rr),
                                                               max(r["best_epoch"] for r in rr)))
        put("rerun_{}_planned_best_epoch".format(k), "{} to {}".format(
            min(r["best_epoch"] for r in rerun[k][2]), max(r["best_epoch"] for r in rerun[k][2])))
        put("rerun_{}_macro_range".format(k), "{} to {}".format(f3(min(r["macro_f1"] for r in rr)),
                                                                f3(max(r["macro_f1"] for r in rr))))
        put("rerun_{}_binary_range".format(k), "{} to {}".format(f3(min(r["binary_f1"] for r in rr)),
                                                                 f3(max(r["binary_f1"] for r in rr))))
    if len(rerun) == 2:
        k_ = int(sum(rerun[k][0].sum() for k in rerun))
        n_ = int(sum(len(rerun[k][0]) for k in rerun))
        put("rerun_b3b4_failed", "{} of {}".format(k_, n_))
        lo, hi = cp_interval(k_, n_)
        put("rerun_b3b4_failed_ci", "{:.0f}% to {:.0f}%".format(100 * lo, 100 * hi))
        put("rerun_b3b4_macro_range", "{} to {}".format(
            f3(min(r["macro_f1"] for k in rerun for r in rerun[k][1])),
            f3(max(r["macro_f1"] for k in rerun for r in rerun[k][1]))))
        put("rerun_b3b4_planned_macro_range", "{} to {}".format(
            f3(min(r["macro_f1"] for k in rerun for r in rerun[k][2])),
            f3(max(r["macro_f1"] for k in rerun for r in rerun[k][2]))))

    # Table 9 over all seeds, and the planned five-seed version for the appendix
    main_rows = [("rf", "Random forest (B1)", runs_of(results, "baseline_random_forest")),
                 ("xgb", "XGBoost (B1)", runs_of(results, "baseline_xgboost"))]
    main_rows += [(k, label, runs_all(results, tags[k])) for k, label in [
        ("mlp", "Flow-only MLP (B2)"), ("b3", "E-GraphSAGE-style GNN (B3)"), ("b4", "Centrality-initialised GNN (B4)"),
        ("concat", "CAG-IDS, concatenation"), ("cagids", "CAG-IDS, gated"), ("gatv2", "CAG-IDS, gated, GATv2"),
        ("c1gate", "C1 + gate control")]]
    metric_keys = [("macro_f1", 1, False), ("macro_f1_attacks", 1, False), ("weighted_f1", 1, False),
                   ("binary_f1", 1, False), ("false_positive_rate", 100, True), ("pr_auc", 1, False)]
    for basis, take in (("all", None), ("five", 5)):
        cells, means = [], []
        for key, label, runs in main_rows:
            rr = runs if take is None else runs[:take]
            row_txt, row_mean = [], []
            for mk, scale, _ in metric_keys:
                vals = np.array([r[mk] for r in rr], float) * scale
                digits = 2 if scale == 100 else 3
                row_txt.append("{:.{d}f} ± {:.{d}f}".format(vals.mean(), vals.std(ddof=1), d=digits))
                row_mean.append(round(vals.mean(), digits))
            cells.append([label] + row_txt)
            means.append(row_mean)
            if basis == "all":
                put("t9_{}_n".format(key), len(rr))
                for (mk, scale, _), txt in zip(metric_keys, row_txt):
                    put("t9_{}_{}".format(key, mk), txt)
        means = np.array(means)
        rows = []
        for i, row in enumerate(cells):
            out = [row[0]]
            for j, (mk, scale, low_best) in enumerate(metric_keys):
                best = means[:, j].min() if low_best else means[:, j].max()
                out.append({"text": row[j + 1], "bold": True} if means[i, j] == best else row[j + 1])
            rows.append(out)
        V["table_main_" + basis] = rows

    # Table 11 (per-class F1) and Table A1 (precision / recall) over all seeds
    pc_models = [("RF", runs_of(results, "baseline_random_forest")), ("XGB", runs_of(results, "baseline_xgboost"))]
    pc_models += [(k, runs_all(results, tags[k])) for k in ("mlp", "b3", "b4", "concat", "cagids", "gatv2")]
    test_support = {c: int(meta["class_counts"]["test"][c]) for c in classes}
    rows = []
    for c in classes:
        vals = [np.mean([r["per_class"][c]["f1"] for r in runs]) for _, runs in pc_models]
        best = max(round(v, 3) for v in vals)
        rows.append([c] + [{"text": f3(v), "bold": True} if round(v, 3) == best else f3(v) for v in vals] +
                    ["{:,}".format(test_support[c])])
        for (name, _), v in zip(pc_models, vals):
            V["_pc_{}_{}".format(name, c)] = v
    V["table_perclass"] = rows
    for name, k in (("RF", "rf"), ("mlp", "mlp"), ("cagids", "cagids"), ("b4", "b4"), ("concat", "concat")):
        for c in ("Analysis", "Worms", "Shellcode", "DoS", "Exploits", "Backdoor"):
            put("pc20_{}_{}".format(k, c.lower()), f3(V["_pc_{}_{}".format(name, c)]))
    a1_models = [("Random forest", runs_of(results, "baseline_random_forest")),
                 ("Flow-only MLP (B2)", runs_all(results, "flow_mlp")), ("CAG-IDS, gated", runs_all(results, "cagids"))]
    rows = []
    for c in ("Analysis", "Backdoor", "DoS", "Shellcode", "Worms"):
        row = [c]
        for name, runs in a1_models:
            pr = np.mean([r["per_class"][c]["precision"] for r in runs])
            rc = np.mean([r["per_class"][c]["recall"] for r in runs])
            row.append("{} / {}".format(f3(pr), f3(rc)))
            put("a1_{}_{}".format(name.split()[0].lower().replace("-", ""), c.lower()), "{} / {}".format(f3(pr), f3(rc)))
        row.append("{:,}".format(test_support[c]))
        rows.append(row)
    V["table_a1"] = rows

    comparisons = [("cagids", "b3", "Gated CAG-IDS − E-GraphSAGE-style (H1)", "cagids_b3"),
                   ("cagids", "b4", "Gated CAG-IDS − centrality-initialised (H1)", "cagids_b4"),
                   ("cagids", "concat", "Gate − concatenation (H3)", "cagids_concat"),
                   ("cagids", "c1gate", "Gated CAG-IDS − C1 + gate control", "cagids_c1gate"),
                   ("c1gate", "b4", "C1 + gate control − centrality-initialised", "c1gate_b4"),
                   ("gatv2", "cagids", "GATv2 − mean aggregation (both gated)", "gatv2_cagids"),
                   ("b3", "mlp", "E-GraphSAGE-style − flow-only MLP", "b3_mlp"),
                   ("b4", "mlp", "Centrality-initialised − flow-only MLP", "b4_mlp"),
                   ("cagids", "mlp", "Gated CAG-IDS − flow-only MLP", "cagids_mlp"),
                   ("cagids", "rf", "Gated CAG-IDS − random forest", "cagids_rf")]
    table = []
    # round 3: robust statistics over seeds, and the three planned comparisons with a family-wise correction
    robust_keys = {"cagids_b3", "cagids_b4", "cagids_concat", "cagids_c1gate", "c1gate_b4"}
    planned_keys = ["cagids_b3", "cagids_b4", "cagids_concat"]
    p_planned = {}
    for a, b, label, key in comparisons:
        if preds.get(a) is None or preds.get(b) is None:
            continue
        res = two_level_bootstrap(preds[a], preds[b], edge_window, n_classes, n_boot=args.n_boot,
                                  extra_stats=("median", "trim20") if key in robust_keys else ())
        five = two_level_bootstrap(preds[a][:5], preds[b][:5], edge_window, n_classes, n_boot=args.n_boot)
        ra, rt = res["all"], res["attacks"]
        if key in robust_keys:
            for s, name in (("median", "median"), ("trim20", "trim")):
                rob = ra["robust"][s]
                put("rob_{}_{}".format(key, name), "{} {}".format(signed(rob["delta"]), interval(*rob["two_level"])))
                put("rob_{}_{}_excludes_zero".format(key, name), excludes_zero(rob["two_level"]))
            hl = hodges_lehmann(V["_f1s_" + a], V["_f1s_" + b])
            put("rob_{}_hl".format(key), "{} [{}, {}]".format(signed(hl["hl"]), signed(hl["lo"]), signed(hl["hi"])))
            put("rob_{}_hl4".format(key), "{} [{}, {}]".format(signed(hl["hl"], 4), signed(hl["lo"], 4),
                                                                signed(hl["hi"], 4)))
            put("rob_{}_hl_excludes_zero".format(key), excludes_zero((hl["lo"], hl["hi"])))
            put("rob_{}_mw_p".format(key), "{:.2f}".format(hl["p"]))
            put("rob_{}_psup".format(key), "{:.2f}".format(hl["p_superior"]))
            fa, fb = V["_reply"].get(a), V["_reply"].get(b)
            if fa is not None and fb is not None and (fa.any() or fb.any()):
                pa = [x for x, f in zip(preds[a], fa) if not f]
                pb = [x for x, f in zip(preds[b], fb) if not f]
                nr = two_level_bootstrap(pa, pb, edge_window, n_classes, n_boot=args.n_boot)
                put("rob_{}_noreply".format(key), ci_text(nr))
                put("rob_{}_noreply_excludes_zero".format(key), excludes_zero(nr["all"]["two_level"]))
                put("rob_{}_noreply_seeds".format(key), "{} vs {}".format(len(pa), len(pb)))
        if key in planned_keys:
            put("tl_{}_bonf".format(key), interval(*ra["two_level_bonf3"]))
            # round 4: four decimals, since a bound that rounds to +0.000 cannot be told apart from zero
            put("tl_{}_bonf4".format(key), interval(*ra["two_level_bonf3"], digits=4))
            put("tl_{}_bonf_excludes_zero".format(key), excludes_zero(ra["two_level_bonf3"]))
            p_le = ra["p_two_level_le_0"]
            p_planned[key] = min(1.0, 2 * min(p_le, 1 - p_le))
            put("tl_{}_p_two_sided".format(key), "{:.3f}".format(p_planned[key]))
            # round 6: the smallest difference these seeds and windows detect with 80% power at the 5% level,
            # and equivalence within SESOI (both 90% bounds inside the margin)
            put("mde_{}".format(key), f3(MDE_Z * ra["two_level_sd"]))
            put("eq90_{}".format(key), interval(*ra["two_level_90"]))
            put("eq90_{}_4".format(key), interval(*ra["two_level_90"], digits=4))
            put("eq_{}_within_sesoi".format(key), bool(-SESOI < ra["two_level_90"][0] and ra["two_level_90"][1] < SESOI))
        put("tl_{}".format(key), ci_text(res))
        put("tl_{}_delta".format(key), signed(ra["delta"]))
        put("tl_{}_ci".format(key), interval(*ra["two_level"]))
        put("tl_{}_window_ci".format(key), interval(*ra["window_only"]))
        put("tl_{}_seed_ci".format(key), interval(*ra["seed_only"]))
        put("tl_{}_att".format(key), ci_text(res, "attacks"))
        put("tl_{}_welch_p".format(key), "{:.3f}".format(ra["welch_p"]))
        put("tl_{}_seeds".format(key), "{} vs {}".format(res["n_seeds_a"], res["n_seeds_b"]))
        put("tl_{}_excludes_zero".format(key), excludes_zero(ra["two_level"]))
        put("tl5_{}".format(key), "{} {}".format(signed(five["all"]["delta"]), interval(*five["all"]["window_only"])))
        table.append([label, signed(five["all"]["delta"]), interval(*five["all"]["window_only"]), signed(ra["delta"]),
                      interval(*ra["two_level"])])
    if len(p_planned) == 3:
        # Holm's step-down adjustment of the bootstrap p-values of the three planned comparisons
        order = sorted(p_planned, key=p_planned.get)
        running = 0.0
        for i, key in enumerate(order):
            running = max(running, min(1.0, (3 - i) * p_planned[key]))
            put("tl_{}_p_holm".format(key), "{:.3f}".format(running))
    # the two random-split rows: models trained and tested under the random flow split. Their seeds 6 to 20
    # are used only when all five neural models have them, so that the random-split results share one seed basis
    rf20 = all((results / "gnn_extra_rf_{}.json".format(t)).exists() for t, _, _ in NEURAL)
    put("rf_all20", "yes" if rf20 else "no")
    rf_preds = (lambda t: preds_all(pred_dir, t)) if rf20 else (lambda t: seed_preds(pred_dir, t))
    rf_runs = (lambda t: runs_all(results, t)) if rf20 else (lambda t: runs_of(results, "gnn_" + t))
    edge_window_random = np.load(work / "graphs_random_flow" / "graphs.npz")["edge_window"]
    for a, b, label, key in [("rf_cagids", "rf_flow_mlp", "Gated CAG-IDS − flow-only MLP, random flow split", "rnd_cagids_mlp"),
                             ("rf_cagids", "rf_centrality_egraphsage",
                              "Gated CAG-IDS − centrality-initialised, random flow split", "rnd_cagids_b4")]:
        pa, pb = rf_preds(a), rf_preds(b)
        res = two_level_bootstrap(pa, pb, edge_window_random, n_classes, n_boot=args.n_boot)
        five = two_level_bootstrap(pa[:5], pb[:5], edge_window_random, n_classes, n_boot=args.n_boot)
        ra = res["all"]
        put("tl_{}".format(key), ci_text(res))
        put("tl_{}_seeds".format(key), "{} vs {}".format(res["n_seeds_a"], res["n_seeds_b"]))
        put("tl_{}_excludes_zero".format(key), excludes_zero(ra["two_level"]))
        table.append([label, signed(five["all"]["delta"]), interval(*five["all"]["window_only"]), signed(ra["delta"]),
                      interval(*ra["two_level"])])
    V["table_two_level"] = table

    if "_f1s_cagids" in V and "_f1s_concat" in V:
        vt = variance_tests(V["_f1s_cagids"], V["_f1s_concat"])
        put("var_gate_concat", vt)
        put("var_gate_sd", f3(vt["sd_a"]))
        put("var_concat_sd", f3(vt["sd_b"]))
        put("var_ratio", "{:.2f}".format(1 / vt["var_ratio"]))
        put("var_p_f", "{:.3f}".format(vt["p_f"]))
        put("var_p_bf", "{:.3f}".format(vt["p_brown_forsythe"]))
        five = variance_tests(V["_f1s_cagids"][:5], V["_f1s_concat"][:5])
        put("var5_ratio", "{:.1f}".format(1 / five["var_ratio"]))
        put("var5_crit", "{:.2f}".format(five["f_crit_975"]))
        put("var5_p_f", "{:.2f}".format(five["p_f"]))
        sds = {k: np.std(V["_f1s_" + k], ddof=1) for k in ("mlp", "b3", "b4", "concat", "cagids", "c1gate", "gatv2")
               if "_f1s_" + k in V}
        put("s20_sd_list", {k: f3(v) for k, v in sds.items()})
        put("s20_sd_min", f3(min(sds.values())))
        put("s20_sd_max", f3(max(sds.values())))
        shifts = {k: abs(np.mean(V["_f1s_" + k]) - np.mean(V["_f1s_" + k][:5])) for k in sds}
        put("s20_max_shift", f3(max(shifts.values())))
        put("s20_max_shift_model", max(shifts, key=shifts.get))
        put("s20_shift_others_max", f3(max(v for k, v in shifts.items() if k != "b4")))
    ctx_only = runs_of(results, "gnn_context_only")
    if ctx_only:
        put("ctxonly_macro", pm(macro_list(ctx_only)))
        put("ctxonly_binary", pm(macro_list(ctx_only, "binary_f1")))
    best_epochs = [r["best_epoch"] for k in ("mlp", "b3", "b4", "concat", "cagids", "gatv2") for r in runs_all(results, tags[k])]
    put("best_epoch_median", int(np.median(best_epochs)))
    put("best_epoch_n", len(best_epochs))
    put("best_epoch_late", int(sum(e > 24 for e in best_epochs)))
    if ctx_only:
        put("ctxonly_attack", pm(macro_list(ctx_only, "macro_f1_attacks")))
        put("ctxonly_fpr", "{:.4f}".format(np.mean(macro_list(ctx_only, "false_positive_rate"))))

    # round 6 (a): B3, B4 and gated CAG-IDS with the longer budget (60 epochs, patience 12) over seeds 1 to 20.
    # Every seed is trained this way, not only those that failed, so the failure rate and the gap under this
    # budget are not selected on the outcome.
    long_tags = {"b3": "egraphsage", "b4": "centrality_egraphsage", "cagids": "cagids"}
    LP = {k: seed_preds(pred_dir, "long_" + t, "extra_long_" + t) for k, t in long_tags.items()}
    if all(p is not None and len(p) == len(ALL_SEEDS) for p in LP.values()):
        lflags = {}
        for k, p in LP.items():
            f1s = [macro_f1_from_pred(y, q, n_classes) for _, y, q in p]
            rr = runs_all(results, "long_" + long_tags[k])
            lflags[k] = reply_flags(p)
            V["_long_f1s_" + k] = f1s
            put("long20_{}_macro".format(k), pm(f1s))
            put("long20_{}_median".format(k), f3(np.median(f1s)))
            put("long20_{}_range".format(k), "{} to {}".format(f3(min(f1s)), f3(max(f1s))))
            put("long20_{}_failed".format(k), "{} of {}".format(int(lflags[k].sum()), len(p)))
            put("long20_{}_gain".format(k), signed(np.mean(f1s) - np.mean(V["_f1s_" + k])))
            put("long20_{}_best_epoch".format(k), "{} to {}".format(min(r["best_epoch"] for r in rr),
                                                                    max(r["best_epoch"] for r in rr)))
            put("long20_{}_best_epoch_median".format(k), int(np.median([r["best_epoch"] for r in rr])))
            # round 9: which seeds fail under the longer budget, and whether they failed under the planned settings
            failed_ids = [i for i, f in enumerate(lflags[k]) if f]
            put("long20_{}_failed_seeds".format(k), ", ".join(str(ALL_SEEDS[i]) for i in failed_ids) or "none")
            if k in V["_reply"]:
                put("long20_{}_failed_seeds_planned_failed".format(k),
                    ", ".join("yes" if V["_reply"][k][i] else "no" for i in failed_ids) or "none")
            shares = []
            for ids, y, q in p:
                m = (y == 0) & to_attacker[ids]
                shares.append(float((q[m] != 0).mean()))
            V["_long_shares_" + k] = shares
            put("long20_{}_binary".format(k), pm(macro_list(rr, "binary_f1")))
            # the seeds that failed under the planned settings, now trained with the longer budget
            planned_failed = V["_reply"][k] if k in V["_reply"] else np.zeros(len(p), bool)
            if planned_failed.any():
                put("long20_{}_planned_failed_now".format(k), "{} of {}".format(
                    int(lflags[k][planned_failed].sum()), int(planned_failed.sum())))
        k_ = int(lflags["b3"].sum() + lflags["b4"].sum())
        lo, hi = cp_interval(k_, 2 * len(ALL_SEEDS))
        put("long20_b3b4_failed", "{} of {}".format(k_, 2 * len(ALL_SEEDS)))
        put("long20_b3b4_failed_ci", "{:.0f}% to {:.0f}%".format(100 * lo, 100 * hi))
        kp = int(V["_reply"]["b3"].sum() + V["_reply"]["b4"].sum())
        put("long20_fisher_p", "{:.1e}".format(stats.fisher_exact([[kp, 40 - kp], [k_, 40 - k_]])[1]))
        for a, b in (("cagids", "b3"), ("cagids", "b4")):
            res = two_level_bootstrap(LP[a], LP[b], edge_window, n_classes, n_boot=args.n_boot)
            key = "long20_{}_{}".format(a, b)
            put(key, ci_text(res))
            put(key + "_att", ci_text(res, "attacks"))
            put(key + "_excludes_zero", excludes_zero(res["all"]["two_level"]))
            put(key + "_mde", f3(MDE_Z * res["all"]["two_level_sd"]))
            put(key + "_eq90", interval(*res["all"]["two_level_90"]))
            five = two_level_bootstrap(LP[a][:5], LP[b][:5], edge_window, n_classes, n_boot=args.n_boot)
            V["_t10_long_" + a + "_" + b] = [signed(five["all"]["delta"]), interval(*five["all"]["window_only"]),
                                             signed(res["all"]["delta"]), interval(*res["all"]["two_level"])]
        for k in LP:
            res = two_level_bootstrap(LP[k], preds[k], edge_window, n_classes, n_boot=args.n_boot)
            put("long20_{}_minus_planned".format(k), ci_text(res))
    else:
        print("longer-budget runs incomplete:", {k: None if p is None else len(p) for k, p in LP.items()})

    # round 6 (b): H1 and H3 after the prior correction of Section 6.8, over 20 seeds. Seeds 6 to 20 choose
    # lambda on their own validation outputs; seeds 1 to 5, whose validation outputs were not saved, use the
    # median lambda of seeds 6 to 20 of the same model, so no lambda is chosen on the test set.
    def test_scores(path, key):
        d = np.load(path)
        order = np.argsort(d["edge_ids"])
        return d["edge_ids"][order].astype("int32"), d["y_true"][order].astype("int8"), d[key][order].astype("float64")

    PC, pc_lam = {}, {}
    for k, t in [("mlp", "flow_mlp"), ("b3", "egraphsage"), ("b4", "centrality_egraphsage"),
                 ("concat", "cagids_concat"), ("cagids", "cagids")]:
        extra, lams, log_ws = [], [], []
        for s in EXTRA_SEEDS:
            path = pred_dir / "extra_{}_seed{}.npz".format(t, s)
            _, vs, vy, ts, ty, log_w = load_correction_inputs(path, "neural")
            r = prior_correction(vs, vy, ts, ty, log_w, n_classes)
            ids = np.sort(np.load(path)["edge_ids"]).astype("int32")
            extra.append((ids, ty.astype("int8"), r["pred_tuned"].astype("int8")))
            lams.append(r["lambda"])
            log_ws.append(log_w)
        assert all(np.allclose(w, log_ws[0]) for w in log_ws), "class weights differ between seeds"
        lam = float(np.median(lams))
        planned, agree = [], []
        for s in PLANNED_SEEDS:
            ids, y, prob = test_scores(pred_dir / "{}_seed{}.npz".format(t, s), "y_prob")
            scores = np.log(np.maximum(prob, 6e-8))   # float16 probabilities: floor at the smallest subnormal
            agree.append(float((scores.argmax(1) == preds[k][s - 1][2]).mean()))
            planned.append((ids, y, (scores - lam * log_ws[0]).argmax(1).astype("int8")))
        PC[k] = planned + extra
        pc_lam[k] = lam
        f1s = [macro_f1_from_pred(y, q, n_classes) for _, y, q in PC[k]]
        V["_pc_f1s_" + k] = f1s
        put("prior20_{}_macro".format(k), pm(f1s))
        put("prior20_{}_lambda".format(k), "{:.1f}".format(lam))
        put("prior20_{}_argmax_agreement".format(k), "{:.4f}".format(min(agree)))
        put("prior20_{}_gain".format(k), signed(np.mean(f1s) - np.mean(V["_f1s_" + k])))
    p_pc = {}
    for a, b, key in (("cagids", "b3", "cagids_b3"), ("cagids", "b4", "cagids_b4"), ("cagids", "concat", "cagids_concat"),
                      ("b4", "b3", "b4_b3"), ("cagids", "mlp", "cagids_mlp")):
        res = two_level_bootstrap(PC[a], PC[b], edge_window, n_classes, n_boot=args.n_boot)
        ra = res["all"]
        put("prior20_{}".format(key), ci_text(res))
        put("prior20_{}_att".format(key), ci_text(res, "attacks"))
        put("prior20_{}_excludes_zero".format(key), excludes_zero(ra["two_level"]))
        five = two_level_bootstrap(PC[a][:5], PC[b][:5], edge_window, n_classes, n_boot=args.n_boot)
        put("prior20_{}_five".format(key), "{} {}".format(signed(five["all"]["delta"]),
                                                          interval(*five["all"]["window_only"])))
        V["_t10_prior_" + key] = [signed(five["all"]["delta"]), interval(*five["all"]["window_only"]),
                                  signed(ra["delta"]), interval(*ra["two_level"])]
        if key in planned_keys:
            put("prior20_{}_bonf4".format(key), interval(*ra["two_level_bonf3"], digits=4))
            put("prior20_{}_mde".format(key), f3(MDE_Z * ra["two_level_sd"]))
            put("prior20_{}_eq90".format(key), interval(*ra["two_level_90"]))
            p_le = ra["p_two_level_le_0"]
            p_pc[key] = min(1.0, 2 * min(p_le, 1 - p_le))
    order = sorted(p_pc, key=p_pc.get)
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (3 - i) * p_pc[key]))
        put("prior20_{}_p_holm".format(key), "{:.3f}".format(running))
    pc_rows = []
    for k, label in [("mlp", "Flow-only MLP (B2)"), ("b3", "E-GraphSAGE-style (B3)"), ("b4", "Centrality-init. (B4)"),
                     ("concat", "CAG-IDS, concat."), ("cagids", "CAG-IDS, gated")]:
        pc_rows.append([label, pm(V["_f1s_" + k]), "{} (λ = {:.1f})".format(pm(V["_pc_f1s_" + k]), pc_lam[k])])
    V["table_pc20"] = pc_rows

    # round 6 (c): paired tests between B4 and gated CAG-IDS in Table 15 (seeds 6 to 20, all training flows),
    # after the tuned prior correction and after per-class offsets
    pair15 = {"tuned": {k: PC[k][len(PLANNED_SEEDS):] for k in ("b4", "cagids")}}
    if not args.skip_corrections:
        rows_c, owners, outs = run_corrections(args, work, results, pred_dir, meta, classes, edge_window)
        offs = {}
        for k, row_key, t in (("b4", "b4_all", "extra_centrality_egraphsage"), ("cagids", "cagids_all", "extra_cagids")):
            biases = [o["bias"] for o, w in zip(outs, owners) if w == row_key]
            offs[k] = []
            for s, bias in zip(EXTRA_SEEDS, biases):
                ids, y, ts = test_scores(pred_dir / "{}_seed{}.npz".format(t, s), "y_logit")
                offs[k].append((ids, y, (ts + np.asarray(bias)).argmax(1).astype("int8")))
        pair15["offsets"] = offs
    for name, P in pair15.items():
        res = two_level_bootstrap(P["cagids"], P["b4"], edge_window, n_classes, n_boot=args.n_boot)
        put("pair15_{}_cagids_b4".format(name), ci_text(res))
        put("pair15_{}_cagids_b4_excludes_zero".format(name), excludes_zero(res["all"]["two_level"]))
        put("pair15_{}_means".format(name), "{} vs {}".format(f3(res["all"]["a_mean"]), f3(res["all"]["b_mean"])))
    # extra rows of Table 10: the planned comparisons under the longer budget and after the prior correction
    t10_extra = []
    for key, label in (("_t10_long_cagids_b3", "Gated CAG-IDS − E-GraphSAGE-style, 60 epochs, patience 12"),
                       ("_t10_long_cagids_b4", "Gated CAG-IDS − centrality-initialised, 60 epochs, patience 12"),
                       ("_t10_prior_cagids_b3", "Gated CAG-IDS − E-GraphSAGE-style, prior correction"),
                       ("_t10_prior_cagids_b4", "Gated CAG-IDS − centrality-initialised, prior correction"),
                       ("_t10_prior_cagids_concat", "Gate − concatenation, prior correction")):
        if key in V:
            t10_extra.append([label] + V[key])
    V["table_two_level_r6"] = t10_extra

    # round 9 (a): the 12 failed seeds were trained twice with the longer budget, once in the rerun of failed seeds
    # and once in the 20-seed run. GPU scatter operations make the two runs of a seed differ; how much?
    for k, tag in (("b3", "failrerun_egraphsage"), ("b4", "failrerun_centrality_egraphsage")):
        rr = runs_of(results, "gnn_" + tag)
        if not rr or "_long_shares_" + k not in V:
            continue
        seeds = [r["seed"] for r in rr]
        first = []
        for ids, y, q in seed_preds(pred_dir, tag, seeds_planned=seeds):
            m = (y == 0) & to_attacker[ids]
            first.append(float((q[m] != 0).mean()))
        second = [V["_long_shares_" + k][ALL_SEEDS.index(s)] for s in seeds]
        f1_first = [r["macro_f1"] for r in rr]
        f1_second = [V["_long_f1s_" + k][ALL_SEEDS.index(s)] for s in seeds]
        put("overlap_{}_seeds".format(k), ", ".join(str(s) for s in seeds))
        put("overlap_{}_share_pairs".format(k), {s: "{:.0f}% / {:.0f}%".format(100 * a, 100 * b)
                                                 for s, a, b in zip(seeds, first, second)})
        put("overlap_{}_max_share_second".format(k), "{:.0f}%".format(100 * max(second)))
        put("overlap_{}_max_share_second_seed".format(k), seeds[int(np.argmax(second))])
        put("overlap_{}_macro_absdiff_max".format(k), f3(max(abs(a - b) for a, b in zip(f1_first, f1_second))))
        put("overlap_{}_macro_absdiff_median".format(k), f3(np.median([abs(a - b) for a, b in zip(f1_first, f1_second)])))

    # round 9 (b): the five neural models trained with square-root weights over seeds 1 to 20; H1 and H3 on models
    # trained without the extreme weights instead of corrected after training
    SW = {k: seed_preds(pred_dir, "sw_" + t, "extra_sw_" + t) for t, k, _ in NEURAL}
    if all(p is not None and len(p) == len(ALL_SEEDS) for p in SW.values()):
        for t, k, _ in NEURAL:
            f1s = [macro_f1_from_pred(y, q, n_classes) for _, y, q in SW[k]]
            V["_sw_f1s_" + k] = f1s
            flags = reply_flags(SW[k])
            put("sw20_{}_macro".format(k), pm(f1s))
            put("sw20_{}_median".format(k), f3(np.median(f1s)))
            put("sw20_{}_failed".format(k), "{} of {}".format(int(flags.sum()), len(flags)))
            put("sw20_{}_gain".format(k), signed(np.mean(f1s) - np.mean(V["_f1s_" + k])))
            rr = runs_all(results, "sw_" + t)
            put("sw20_{}_best_epoch_median".format(k), int(np.median([r["best_epoch"] for r in rr])))
        k_ = sum(int(reply_flags(SW[k]).sum()) for k in ("b3", "b4"))
        lo, hi = cp_interval(k_, 2 * len(ALL_SEEDS))
        put("sw20_b3b4_failed", "{} of {}".format(k_, 2 * len(ALL_SEEDS)))
        put("sw20_b3b4_failed_ci", "{:.0f}% to {:.0f}%".format(100 * lo, 100 * hi))
        p_sw = {}
        for a, b, key in (("cagids", "b3", "cagids_b3"), ("cagids", "b4", "cagids_b4"), ("cagids", "concat", "cagids_concat"),
                          ("cagids", "mlp", "cagids_mlp"), ("b4", "b3", "b4_b3"), ("concat", "b3", "concat_b3")):
            res = two_level_bootstrap(SW[a], SW[b], edge_window, n_classes, n_boot=args.n_boot)
            ra = res["all"]
            put("sw20_{}".format(key), ci_text(res))
            put("sw20_{}_att".format(key), ci_text(res, "attacks"))
            put("sw20_{}_excludes_zero".format(key), excludes_zero(ra["two_level"]))
            put("sw20_{}_mde".format(key), f3(MDE_Z * ra["two_level_sd"]))
            put("sw20_{}_eq90".format(key), interval(*ra["two_level_90"]))
            if key in planned_keys:
                put("sw20_{}_bonf4".format(key), interval(*ra["two_level_bonf3"], digits=4))
                p_le = ra["p_two_level_le_0"]
                p_sw[key] = min(1.0, 2 * min(p_le, 1 - p_le))
        order = sorted(p_sw, key=p_sw.get)
        running = 0.0
        for i, key in enumerate(order):
            running = max(running, min(1.0, (3 - i) * p_sw[key]))
            put("sw20_{}_p_holm".format(key), "{:.3f}".format(running))
    else:
        print("square-root-weight runs incomplete:", {k: None if p is None else len(p) for k, p in SW.items()})

    # round 9 (c): Table 10 split into the planned comparisons and an exploratory block without five-seed columns
    explo = []
    for key, label in (("long20_cagids_b3", "Gated CAG-IDS − E-GraphSAGE-style, 60 epochs, patience 12"),
                       ("long20_cagids_b4", "Gated CAG-IDS − centrality-initialised, 60 epochs, patience 12"),
                       ("sw20_cagids_b3", "Gated CAG-IDS − E-GraphSAGE-style, √-inverse weights"),
                       ("sw20_cagids_b4", "Gated CAG-IDS − centrality-initialised, √-inverse weights"),
                       ("sw20_cagids_concat", "Gate − concatenation, √-inverse weights"),
                       ("prior20_cagids_b3", "Gated CAG-IDS − E-GraphSAGE-style, prior correction"),
                       ("prior20_cagids_b4", "Gated CAG-IDS − centrality-initialised, prior correction"),
                       ("prior20_cagids_concat", "Gate − concatenation, prior correction")):
        if key in V:
            d_, iv = V[key].split(" ", 1)
            explo.append([label, "–", "–", d_, iv])
    V["table_two_level_r9"] = explo

    # ------------------------------------------------------------------ 2
    print("\n=== 2. the forest's recipe, a prior correction and per-class offsets ===")
    forest_runs = runs_of(results, "baseline_random_forest")
    put("rf_macro", pm(macro_list(forest_runs)))
    for tag, key, label in NEURAL:
        sub = runs_of(results, "gnn_sub_" + tag)
        if sub:
            full = runs_all(results, tag)
            put("sub_{}_macro".format(key), pm(macro_list(sub)))
            put("sub_{}_attack".format(key), pm(macro_list(sub, "macro_f1_attacks")))
            put("sub_{}_gain".format(key), signed(np.mean(macro_list(sub)) - np.mean(macro_list(full[:5]))))
            put("sub_{}_gain20".format(key), signed(np.mean(macro_list(sub)) - np.mean(macro_list(full))))
            put("sub_{}_gap_rf".format(key), signed(np.mean(macro_list(sub)) - np.mean(macro_list(forest_runs))))
            for c in ("Worms", "Shellcode", "Analysis", "DoS"):
                pr = np.mean([r["per_class"][c]["precision"] for r in sub])
                rc = np.mean([r["per_class"][c]["recall"] for r in sub])
                put("sub_{}_{}_pr".format(key, c.lower()), "{} / {}".format(f3(pr), f3(rc)))
    for t, k, _ in NEURAL:
        p = seed_preds(pred_dir, "sub_" + t)
        if p is None or preds.get("rf") is None:
            continue
        res = two_level_bootstrap(p, preds["rf"], edge_window, n_classes, n_boot=args.n_boot)
        put("sub_{}_minus_rf".format(k), ci_text(res))
    del preds

    if args.skip_corrections:
        correction_rows, owners, outs = CORRECTION_ROWS, [], []
    else:
        correction_rows, owners, outs = run_corrections(args, work, results, pred_dir, meta, classes, edge_window)
    put("val_windows", int(len(np.unique(edge_window[np.flatnonzero(np.load(work / "graphs" / "edge_split.npy") == 1)]))))
    put("val_worms", int(meta["class_counts"]["val"]["Worms"]))
    put("val_backdoor", int(meta["class_counts"]["val"]["Backdoor"]))
    by_row = {}
    for key, o in zip(owners, outs):
        if o is not None:
            by_row.setdefault(key, []).append(o)
    boot_means = {}
    boot_by_seed = {}
    corr_rows = []
    for key, label, tag, seeds, kind in correction_rows:
        rows = by_row.get(key)
        if not rows or len(rows) != len(seeds):
            print("no validation outputs for", tag)
            continue
        plain = [r["plain"] for r in rows]
        full_c = [r["full"] for r in rows]
        tuned = [r["tuned"] for r in rows]
        lams = [r["lambda"] for r in rows]
        offs = [r["offsets"] for r in rows]
        # offsets refitted on each validation resample, for the first five seeds; the trained
        # models stay fixed, so the spread shows how much the offsets depend on the validation sample
        bo = np.array([r["boot_offsets"] for r in rows if r["boot_offsets"]])   # (5, B)
        mo = bo.mean(axis=0)
        boot_means[key] = mo
        boot_by_seed[key] = bo
        # round 3: the same refits with the five seeds resampled as well (20 seed draws per validation resample)
        rng_s = np.random.default_rng(7)
        two = [bo[rng_s.integers(0, len(bo), len(bo)), b].mean() for b in range(bo.shape[1]) for _ in range(20)]
        put("pcb_{}_offsets_ci2".format(key), interval(*np.percentile(two, [2.5, 97.5])).replace("+", ""))
        base5 = np.mean(offs[:5])
        put("pc_{}_plain".format(tag), pm(plain))
        put("pc_{}_full".format(tag), pm(full_c))
        put("pc_{}_tuned".format(tag), pm(tuned))
        put("pc_{}_lambda".format(tag), "{:.1f} (range {:.1f} to {:.1f})".format(np.median(lams), min(lams), max(lams)))
        put("pc_{}_bias".format(tag), pm(offs))
        put("pc_{}_gain".format(tag), signed(np.mean(tuned) - np.mean(plain)))
        put("pc_{}_worms_pr".format(tag), "{} / {}".format(f3(np.mean([r["worms_pr"][0] for r in rows])),
                                                          f3(np.mean([r["worms_pr"][1] for r in rows]))))
        put("pcb_{}_offsets5".format(key), f3(base5))
        put("pcb_{}_offsets_ci".format(key), interval(*np.percentile(mo, [2.5, 97.5])).replace("+", ""))
        put("pcb_{}_offsets_model_sd".format(key), f3(np.median(bo.std(axis=1, ddof=1))))
        put("pcb_{}_offsets_gain".format(key), "{} {}".format(
            signed(base5 - np.mean(plain[:5])), interval(*np.percentile(mo - np.mean(plain[:5]), [2.5, 97.5]))))
        corr_rows.append([label, str(len(rows)), pm(plain), pm(full_c), "{:.1f}".format(np.median(lams)), pm(tuned),
                          pm(offs), interval(*np.percentile(mo, [2.5, 97.5])).replace("+", "")])
    V["table_prior_correction"] = corr_rows
    if "mlp_sub" in boot_means and "rf" in boot_means:
        # the same validation resamples for both, so the difference is paired over them
        d_off = boot_means["mlp_sub"] - boot_means["rf"]
        rf_plain = np.mean(macro_list(forest_runs))
        put("pcb_best_net_minus_rf_corrected", "{} {}".format(
            signed(np.mean([r["offsets"] for r in by_row["mlp_sub"]]) - np.mean([r["offsets"] for r in by_row["rf"]])),
            interval(*np.percentile(d_off, [2.5, 97.5]))))
        put("pcb_best_net_minus_rf_plain", "{} {}".format(
            signed(np.mean([r["offsets"] for r in by_row["mlp_sub"]]) - rf_plain),
            interval(*np.percentile(boot_means["mlp_sub"] - rf_plain, [2.5, 97.5]))))
        put("pcb_n_val_boot", args.n_val_boot)
        # round 3: the same differences with the seeds of both models resampled as well
        bm, br = boot_by_seed["mlp_sub"], boot_by_seed["rf"]
        rf_seeds = np.array(macro_list(forest_runs))
        rng_s = np.random.default_rng(8)
        d_corr, d_plain = [], []
        for b in range(bm.shape[1]):
            for _ in range(20):
                net = bm[rng_s.integers(0, len(bm), len(bm)), b].mean()
                d_corr.append(net - br[rng_s.integers(0, len(br), len(br)), b].mean())
                d_plain.append(net - rf_seeds[rng_s.integers(0, len(rf_seeds), len(rf_seeds))].mean())
        put("pcb_best_net_minus_rf_corrected_seeds", "{} {}".format(
            signed(np.mean([r["offsets"] for r in by_row["mlp_sub"]]) - np.mean([r["offsets"] for r in by_row["rf"]])),
            interval(*np.percentile(d_corr, [2.5, 97.5]))))
        put("pcb_best_net_minus_rf_plain_seeds", "{} {}".format(
            signed(np.mean([r["offsets"] for r in by_row["mlp_sub"]]) - rf_plain),
            interval(*np.percentile(d_plain, [2.5, 97.5]))))
    # round 3: which corrected network would validation alone pick? (the text names the best on test)
    if outs:
        k_job, val_rows = 0, {}
        for key, label, tag, seeds, kind in correction_rows:
            vals = []
            for s in seeds:
                o = outs[k_job]
                k_job += 1
                if o is None:
                    continue
                loaded = load_correction_inputs(pred_dir / "{}_seed{}.npz".format(tag, s), kind)
                vs, vy = loaded[1], loaded[2]
                vals.append(macro_f1_from_pred(vy, (vs + np.asarray(o["bias"])).argmax(1), n_classes))
            if vals:
                val_rows[key] = vals
                put("pcv_{}_val_offsets".format(key), pm(vals))
        nets = {k: np.mean(v) for k, v in val_rows.items() if k != "rf"}
        if nets:
            best_val = max(nets, key=nets.get)
            best_test = max(nets, key=lambda k: np.mean([r["offsets"] for r in by_row[k]]))
            put("pcv_best_by_val", best_val)
            put("pcv_best_by_test", best_test)
            put("pcv_best_by_val_test", f3(np.mean([r["offsets"] for r in by_row[best_val]])))
            if "rf" in by_row:
                put("pcv_best_by_val_minus_rf_corrected", signed(np.mean([r["offsets"] for r in by_row[best_val]]) -
                                                                np.mean([r["offsets"] for r in by_row["rf"]])))

    # ------------------------------------------------------------------ 3
    print("\n=== 3. flows that end after their window closes ===")
    df = pd.read_parquet(work / "flows.parquet", columns=["start_ms", "FLOW_DURATION_MILLISECONDS", "window_id", "attack"])
    t0 = df["start_ms"].min()
    wend = t0 + (df["window_id"].to_numpy() + 1) * 60000.0
    end = df["start_ms"].to_numpy() + df["FLOW_DURATION_MILLISECONDS"].to_numpy()
    late = end > wend
    excess = (end - wend)[late] / 1000.0
    put("late_share", "{:.2f}%".format(100 * late.mean()))
    put("late_share_attack", "{:.2f}%".format(100 * late[df["attack"].to_numpy() != "Benign"].mean()))
    put("late_n", "{:,}".format(int(late.sum())))
    put("late_excess_median_s", "{:.1f}".format(np.median(excess)))
    put("late_excess_p95_s", "{:.0f}".format(np.percentile(excess, 95)))
    put("late_excess_max_s", "{:.0f}".format(excess.max()))
    put("duration_max_s", "{:.0f}".format(df["FLOW_DURATION_MILLISECONDS"].max() / 1000.0))
    del df

    # ------------------------------------------------------------------ 4
    print("\n=== 4. context groups under both splits; split protocols on shared flows ===")
    split_chrono = np.load(work / "graphs" / "edge_split.npy")
    split_rf = np.load(work / "graphs_random_flow" / "edge_split.npy")
    common_rf = np.flatnonzero((split_chrono == 2) & (split_rf == 2))
    abl_def = [("full", "Full context (C1 + C2 + C3)", "cagids", "rf_cagids"),
               ("no_c1", "without C1 (structural)", "abl_no_structural", "rf_abl_no_structural"),
               ("no_c2", "without C2 (behavioural)", "abl_no_behavioural", "rf_abl_no_behavioural"),
               ("no_c3", "without C3 (role)", "abl_no_role", "rf_abl_no_role"),
               ("c1_only", "C1 only (C1 + gate control)", "c1_gate", "rf_c1_gate"),
               ("no_age", "without the age feature", "noage_cagids", "rf_noage_cagids")]
    abl_def = [d for d in abl_def if runs_of(results, "gnn_" + d[2]) and runs_of(results, "gnn_" + d[3])]
    # round 3: every row uses all of its seeds (20 for the full context, without C2 and without age; the
    # Seeds column gives the counts), and the changes resample the seeds of every model
    AP = {key: (preds_all(pred_dir, ct), preds_all(pred_dir, rt)) for key, _, ct, rt in abl_def}
    abl_rows = []
    for key, label, ct, rt in abl_def:
        rc, rr = runs_all(results, ct), runs_all(results, rt)
        pc_, pr_ = AP[key]
        if len(pc_) != len(rc) or len(pr_) != len(rr):
            raise ValueError("predictions and runs differ for " + key)
        same = contrast([pr_, pc_], [1, -1], edge_window, n_classes, keep_ids=common_rf, n_boot=args.n_boot)
        gap = np.mean(macro_list(rr)) - np.mean(macro_list(rc))
        put("ablrf_{}_n".format(key), "{} / {}".format(len(rr), len(rc)))
        put("ablrf_{}_random".format(key), pm(macro_list(rr)))
        put("ablrf_{}_chrono".format(key), pm(macro_list(rc)))
        put("ablrf_{}_gap".format(key), signed(gap))
        put("ablrf_{}_same".format(key), ci_text(same))
        V["_gap_" + key] = (macro_list(rr), macro_list(rc))
        change = ""
        if key != "full":
            fr, fc = AP["full"][1], AP["full"][0]
            dd = contrast([pr_, pc_, fr, fc], [1, -1, -1, 1], edge_window, n_classes, keep_ids=common_rf, n_boot=args.n_boot)
            change = ci_text(dd)
            put("ablrf_{}_change".format(key), change)
            put("ablrf_{}_change_excludes_zero".format(key), excludes_zero(dd["all"]["two_level"]))
            # the same change on each model's own test sets, with seed noise only
            a_r, a_c = V["_gap_" + key]
            f_r, f_c = V["_gap_full"]
            d = (np.mean(a_r) - np.mean(a_c)) - (np.mean(f_r) - np.mean(f_c))
            se = np.sqrt(sum(np.var(x, ddof=1) / len(x) for x in (a_r, a_c, f_r, f_c)))
            put("ablrf_{}_gapchange_own".format(key), "{} [{}, {}]".format(signed(d), signed(d - 1.96 * se), signed(d + 1.96 * se)))
        seeds_txt = str(len(rr)) if len(rr) == len(rc) else "{} / {}".format(len(rr), len(rc))
        abl_rows.append([label, seeds_txt, pm(macro_list(rr)), pm(macro_list(rc)), signed(gap), ci_text(same),
                         change or "–"])
    V["table_ablation_random"] = abl_rows
    # C1 alone against the model without C2: does C2 add to the gain beyond C1?
    dd = contrast([AP["no_c2"][1], AP["no_c2"][0], AP["c1_only"][1], AP["c1_only"][0]], [1, -1, -1, 1],
                  edge_window, n_classes, keep_ids=common_rf, n_boot=args.n_boot)
    put("ablrf_noc2_minus_c1only", ci_text(dd))
    # round 3: removing age (or all of C2) under each split on its own test flows
    for key in ("no_age", "no_c2"):
        if key in AP:
            r_rnd = two_level_bootstrap(AP[key][1], AP["full"][1], edge_window_random, n_classes, n_boot=args.n_boot)
            r_chr = two_level_bootstrap(AP[key][0], AP["full"][0], edge_window, n_classes, n_boot=args.n_boot)
            put("ablrf_{}_minus_full_random".format(key), ci_text(r_rnd))
            put("ablrf_{}_minus_full_chrono".format(key), ci_text(r_chr))
    del AP

    split_rw = np.load(work / "graphs_random_window" / "edge_split.npy")
    common_rw = np.flatnonzero((split_chrono == 2) & (split_rw == 2))
    lab = np.load(work / "graphs" / "labels.npy")
    put("rw_common_flows", "{:,}".format(len(common_rw)))
    put("rw_common_attack", "{:,}".format(int((lab[common_rw] > 0).sum())))
    put("rw_common_windows", "{:,}".format(len(np.unique(edge_window[common_rw]))))
    put("rf_common_flows", "{:,}".format(len(common_rf)))
    put("rf_common_attack", "{:,}".format(int((lab[common_rf] > 0).sum())))
    put("rf_common_windows", "{:,}".format(len(np.unique(edge_window[common_rf]))))
    rw_test = np.flatnonzero(split_rw == 2)
    put("rw_test_flows", "{:,}".format(len(rw_test)))
    put("rw_test_attack", "{:,}".format(int((lab[rw_test] > 0).sum())))

    ss_runs = {}
    for key, t in [("rf", "baseline_random_forest_session_wv"), ("xgb", "baseline_xgboost_session_wv")]:
        ss_runs[key] = runs_of(results, t)
    for t, k, _ in NEURAL:
        ss_runs[k] = runs_all(results, "swv_" + t)
    model_def = [("Random forest", "rf", "random_forest", "random_forest_random_flow", "random_forest_random_window",
                  "baseline_random_forest", True),
                 ("XGBoost", "xgb", "xgboost", "xgboost_random_flow", "xgboost_random_window", "baseline_xgboost", True)]
    model_def += [(label, k, t, "rf_" + t, "rw_" + t, None, False) for t, k, label in NEURAL]
    SP = {}
    protocol_rows, shared_rows = [], []
    for label, k, t_chrono, t_rf, t_rw, base, flat in model_def:
        if flat:
            rc, rr, rw = runs_of(results, base), runs_of(results, base + "_random_flow"), runs_of(results, base + "_random_window")
            pc_, pr_, pw_ = (seed_preds(pred_dir, t_chrono), seed_preds(pred_dir, t_rf), seed_preds(pred_dir, t_rw))
        else:
            rc, rr, rw = runs_all(results, t_chrono), rf_runs(t_rf), runs_all(results, t_rw)
            pc_, pr_, pw_ = preds_all(pred_dir, t_chrono), rf_preds(t_rf), preds_all(pred_dir, t_rw)
        SP[k] = (pc_, pr_, pw_)
        s_rf = contrast([pr_, pc_], [1, -1], edge_window, n_classes, keep_ids=common_rf, n_boot=args.n_boot)
        s_rw = contrast([pw_, pc_], [1, -1], edge_window, n_classes, keep_ids=common_rw, n_boot=args.n_boot)
        gap_rf = np.mean(macro_list(rr)) - np.mean(macro_list(rc))
        gap_rw = np.mean(macro_list(rw)) - np.mean(macro_list(rc))
        put("sp_{}_seeds".format(k), "{}/{}/{}".format(len(rr), len(rw), len(rc)))
        put("sp_{}_rf".format(k), pm(macro_list(rr)))
        put("sp_{}_rw".format(k), pm(macro_list(rw)))
        put("sp_{}_chrono".format(k), pm(macro_list(rc)))
        put("sp_{}_gap_rf".format(k), signed(gap_rf))
        put("sp_{}_gap_rw".format(k), signed(gap_rw))
        put("sp_{}_same_rf".format(k), ci_text(s_rf))
        put("sp_{}_same_rw".format(k), ci_text(s_rw))
        V["_same_rf_" + k] = s_rf["all"]["delta"]
        V["_gap_rf_" + k] = gap_rf
        ss = ss_runs.get(k)
        protocol_rows.append([label, pm(macro_list(rr)), pm(macro_list(rw)), pm(macro_list(rc)),
                              pm(macro_list(ss)) if ss else "–"])
        shared_rows.append([label, signed(s_rf["all"]["delta"]), interval(*s_rf["all"]["two_level"]),
                            signed(s_rw["all"]["delta"]), interval(*s_rw["all"]["two_level"])])
    V["table_protocols"] = protocol_rows
    V["table_shared"] = shared_rows
    gains_rf = [V["_gap_rf_" + k] for _, k, _ in NEURAL if k not in CONTEXT_MODELS]
    put("sp_gap_rf_graph_range", "{} to {}".format(signed(min(V["_gap_rf_" + k] for k in ("b3", "b4", "concat", "cagids"))),
                                                   signed(max(V["_gap_rf_" + k] for k in ("b3", "b4", "concat", "cagids")))))
    put("sp_gap_rf_others_mean", signed(np.mean(gains_rf)))
    # difference between the gain of the two context-aware models and that of the other three neural models
    for split_key, col, keep in (("rf", 1, common_rf), ("rw", 2, common_rw)):
        groups, coefs = [], []
        for _, k, _ in NEURAL:
            wgt = 0.5 if k in CONTEXT_MODELS else -1.0 / 3.0
            groups += [SP[k][col], SP[k][0]]
            coefs += [wgt, -wgt]
        dd = contrast(groups, coefs, edge_window, n_classes, keep_ids=keep, n_boot=args.n_boot)
        put("did_{}".format(split_key), ci_text(dd))
        put("did_{}_att".format(split_key), ci_text(dd, "attacks"))
        put("did_{}_excludes_zero".format(split_key), excludes_zero(dd["all"]["two_level"]))
        put("did_{}_p_le0".format(split_key), "{:.3f}".format(dd["all"]["p_two_level_le_0"]))
        put("did_{}_seeds".format(split_key), dd["n_seeds"])
        # each context-aware model against the mean of the other three
        for kc in CONTEXT_MODELS:
            groups, coefs = [SP[kc][col], SP[kc][0]], [1.0, -1.0]
            for _, k, _ in NEURAL:
                if k not in CONTEXT_MODELS:
                    groups += [SP[k][col], SP[k][0]]
                    coefs += [-1.0 / 3.0, 1.0 / 3.0]
            d1 = contrast(groups, coefs, edge_window, n_classes, keep_ids=keep, n_boot=args.n_boot)
            put("did_{}_{}".format(split_key, kc), ci_text(d1))
    ctw = load_json(results, "context_probe_tests_random_window")
    if ctw:
        a = ctw["random_window/flow_only"]["macro_f1"]["mean"]
        b = ctw["random_window/flow_plus_context"]["macro_f1"]["mean"]
        put("ctw_flow", f3(a))
        put("ctw_context", f3(b))
        put("ctw_delta", signed(b - a))

    # ------------------------------------------------------------------ 5
    print("\n=== 5. window length with equal mini-batches and aligned periods ===")
    steps = {}
    for tag, wpb in [("graphs_w30", 8), ("graphs", 8), ("graphs_w300", 8), ("graphs_w30_aligned", 16),
                     ("graphs_w300_aligned", 2)]:
        m = json.load(open(work / tag / "meta.json", encoding="utf-8")) if (work / tag / "meta.json").exists() else None
        if not m:
            continue
        n_train = m["split_windows"]["train"]
        flows_train = sum(m["class_counts"]["train"].values())
        steps[tag] = int(np.ceil(n_train / wpb))
        put("steps_{}".format(tag), steps[tag])
        put("flows_per_batch_{}".format(tag), "{:,}".format(int(round(flows_train / n_train * wpb))))
    wl_rows = []
    for label, pattern in [("30 s, as planned (8 windows per batch)", "w30_{}"),
                           ("30 s, 16 windows per batch, 60-s periods", "wa30_{}"),
                           ("60 s, planned", "{}"),
                           ("300 s, as planned (8 windows per batch)", "w300_{}"),
                           ("300 s, 2 windows per batch, 60-s periods", "wa300_{}")]:
        row = [label]
        for t, k, _ in NEURAL:
            # round 3: the planned 60-s row gives the 20-seed means of Table 9
            r = runs_all(results, t) if pattern == "{}" else runs_of(results, "gnn_" + pattern.format(t))
            row.append(pm(macro_list(r)))
            put("wl_{}_{}".format(pattern.split("_")[0] if "_" in pattern else "w60", k), pm(macro_list(r)))
        wl_rows.append(row)
    V["table_window_length"] = wl_rows
    for t, k, _ in NEURAL:
        a = macro_list(runs_of(results, "gnn_wa300_" + t))
        b20 = macro_list(runs_all(results, t))
        b5 = b20[:5]
        d, lo, hi = welch_interval(a, b20)
        put("wa300_{}_vs20".format(k), "{} [{}, {}]".format(signed(d), signed(lo), signed(hi)))
        put("wa300_{}_vs20_excludes_zero".format(k), excludes_zero((lo, hi)))
        d5, lo5, hi5 = welch_interval(a, b5)
        put("wa300_{}_vs5".format(k), "{} [{}, {}]".format(signed(d5), signed(lo5), signed(hi5)))
        a30 = macro_list(runs_of(results, "gnn_wa30_" + t))
        d3, lo3, hi3 = welch_interval(a30, b20)
        put("wa30_{}_vs20".format(k), "{} [{}, {}]".format(signed(d3), signed(lo3), signed(hi3)))
        put("wa30_{}_vs20_excludes_zero".format(k), excludes_zero((lo3, hi3)))
    patterns = ["w30_{}", "wa30_{}", "{}", "w300_{}", "wa300_{}", "sw_{}", "focal_{}", "long_{}"]
    sens = {}
    for pat in patterns:
        for t, k, _ in NEURAL:
            name = "cagids_focal" if (pat == "focal_{}" and t == "cagids") else pat.format(t)
            r = runs_all(results, name) if pat == "{}" else runs_of(results, "gnn_" + name)
            if r:
                sens[(pat, k)] = np.mean(macro_list(r))
    full_rows = [p for p in patterns if all((p, k) in sens for _, k, _ in NEURAL)]
    put("sens_rows_total", len(full_rows))
    put("sens_rows_mlp_best", sum(max(NEURAL, key=lambda x: sens[(p, x[1])])[1] == "mlp" for p in full_rows))
    put("sens_rows_graph_above_mlp", sum(any(sens[(p, k)] > sens[(p, "mlp")] for k in ("b3", "b4", "concat", "cagids"))
                                         for p in full_rows))
    put("sens_rows_cagids_above_b3", sum(sens[(p, "cagids")] > sens[(p, "b3")] for p in full_rows))
    put("sens_rows_b4_above_cagids", sum(sens[(p, "b4")] > sens[(p, "cagids")] for p in full_rows))
    for p in ("wa30_{}", "wa300_{}"):
        base = "w300_{}" if p.startswith("wa300") else "w30_{}"
        d_new = [sens[(p, k)] - sens[("{}", k)] for _, k, _ in NEURAL]
        d_old = [sens[(base, k)] - sens[("{}", k)] for _, k, _ in NEURAL]
        put("wl_{}_change_new".format(p[:-3]), "{} to {}".format(signed(min(d_new)), signed(max(d_new))))
        put("wl_{}_change_old".format(p[:-3]), "{} to {}".format(signed(min(d_old)), signed(max(d_old))))
        put("wl_{}_mlp_change".format(p[:-3]), signed(sens[(p, "mlp")] - sens[("{}", "mlp")]))

    # ------------------------------------------------------------------ 6
    print("\n=== 6. possible machine fingerprints; session split ===")
    diag = load_json(results, "diagnostics_graphs")
    sf = diag["single_feature_probes"]
    for f in FINGERPRINT_FIELDS:
        put("fp_single_{}".format(f), "macro {} binary {}".format(f3(sf[f]["macro_f1"]), f3(sf[f]["binary_f1"])))
    fpp = load_json(results, "fingerprint_probe_graphs")
    lp, hp = fpp["label_probe"], fpp["host_probe"]
    put("fp_joint_macro", f3(lp["macro_f1"]))
    put("fp_joint_attack", f3(lp["macro_f1_attacks"]))
    put("fp_joint_binary", f3(lp["binary_f1"]))
    put("fp_joint_fpr", "{:.4f}".format(lp["false_positive_rate"]))
    put("fp_host_accuracy", "{:.1f}%".format(100 * hp["accuracy"]))
    put("fp_host_majority", "{:.1f}%".format(100 * hp["majority_accuracy"]))
    put("fp_host_balanced_acc", "{:.1f}%".format(100 * hp["balanced_accuracy"]))
    put("fp_n_hosts_test", hp["n_sources_test"])
    for name, base, nofp in [("rf", "baseline_random_forest", "baseline_random_forest_nofp"),
                             ("mlp", "gnn_flow_mlp", "gnn_nofp_flow_mlp"), ("cagids", "gnn_cagids", "gnn_nofp_cagids")]:
        a, b = runs_of(results, base), runs_of(results, nofp)
        if a and b:
            put("nofp_{}_macro".format(name), pm(macro_list(b)))
            put("nofp_{}_delta".format(name), signed(np.mean(macro_list(b)) - np.mean(macro_list(a))))

    # session split: validation on session-1 flows (as first run) and on whole session-1 windows
    for tag_graphs, label in (("graphs_session", "flowval"), ("graphs_session_wv", "winval")):
        m = json.load(open(work / tag_graphs / "meta.json", encoding="utf-8"))
        cc = m["class_counts"]
        put("ss_{}_val_windows".format(label), m["split_windows"]["val"])
        put("ss_{}_train_windows".format(label), m["split_windows"]["train"])
        put("ss_{}_train_attack".format(label), "{:,}".format(sum(v for c, v in cc["train"].items() if c != "Benign")))
        put("ss_{}_val_attack".format(label), "{:,}".format(sum(v for c, v in cc["val"].items() if c != "Benign")))
        put("ss_{}_val_counts".format(label), {c: cc["val"][c] for c in classes})
        put("ss_{}_train_counts".format(label), {c: cc["train"][c] for c in classes})
    cc_ss = json.load(open(work / "graphs_session_wv" / "meta.json", encoding="utf-8"))["class_counts"]
    put("ss_test_flows", "{:,}".format(sum(cc_ss["test"].values())))
    put("ss_test_attack", "{:,}".format(sum(v for c, v in cc_ss["test"].items() if c != "Benign")))
    session_def = [("mlp", "flow_mlp"), ("b3", "egraphsage"), ("b4", "centrality_egraphsage"),
                   ("concat", "cagids_concat"), ("cagids", "cagids"), ("nob", "abl_no_behavioural"),
                   ("noage", "noage_cagids"), ("clip", "clipage_cagids"),
                   ("noage_concat", "noage_cagids_concat"), ("clip_concat", "clipage_cagids_concat"),
                   # round 3: age over active windows only, or capped at one hour; early stopping without
                   # Backdoor and Worms
                   ("active", "activeage_cagids"), ("active_concat", "activeage_cagids_concat"),
                   ("cap", "agecap_cagids"), ("cap_concat", "agecap_cagids_concat")]
    session_def += [("esx_" + k, "esx_" + t) for k, t in session_def[:7] + [("noage_concat", "noage_cagids_concat")]]
    SS = {}
    for k, t in session_def:
        new = runs_all(results, "swv_" + t)
        old = runs_of(results, "gnn_ss_" + t)
        if new:
            SS[k] = preds_all(pred_dir, "swv_" + t)
            put("swv_{}_n".format(k), len(new))
            put("swv_{}_macro".format(k), pm(macro_list(new)))
            put("swv_{}_attack".format(k), pm(macro_list(new, "macro_f1_attacks")))
            put("swv_{}_binary".format(k), pm(macro_list(new, "binary_f1")))
            put("swv_{}_val".format(k), pm(macro_list(new, "best_val_macro_f1")))
            put("swv_{}_min".format(k), f3(min(macro_list(new))))
            put("swv_{}_best_epoch".format(k), "{} to {}".format(min(r["best_epoch"] for r in new),
                                                                 max(r["best_epoch"] for r in new)))
            for c in ("Backdoor", "Worms"):
                put("swv_{}_f1_{}".format(k, c.lower()), f3(np.mean([r["per_class"][c]["f1"] for r in new])))
            if "best_val_early_stop_score" in new[0]:
                put("swv_{}_val_es".format(k), pm(macro_list(new, "best_val_early_stop_score")))
        if old:
            put("ssf_{}_macro".format(k), pm(macro_list(old)))
            put("ssf_{}_val".format(k), pm(macro_list(old, "best_val_macro_f1")))
            put("ssf_{}_binary".format(k), pm(macro_list(old, "binary_f1")))
    for key, t in (("rf", "baseline_random_forest_session"), ("xgb", "baseline_xgboost_session"),
                   ("rf_wv", "baseline_random_forest_session_wv"), ("xgb_wv", "baseline_xgboost_session_wv")):
        r = runs_of(results, t)
        if r:
            put("ssb_{}_macro".format(key), pm(macro_list(r)))
    ew_ss = np.load(work / "graphs_session_wv" / "graphs.npz")["edge_window"]
    session_pairs = [("cagids", "mlp"), ("concat", "mlp"), ("b4", "mlp"), ("b3", "mlp"), ("nob", "mlp"),
                     ("noage", "mlp"), ("clip", "mlp"), ("noage_concat", "mlp"), ("clip_concat", "mlp"),
                     ("noage", "cagids"), ("clip", "cagids"), ("nob", "cagids"), ("noage_concat", "concat"),
                     ("clip_concat", "concat"), ("nob", "noage"), ("nob", "clip"), ("noage", "b4"), ("nob", "b4"),
                     ("cagids", "b4"), ("clip", "noage"),
                     # round 3
                     ("active", "mlp"), ("active", "cagids"), ("active", "noage"), ("active_concat", "mlp"),
                     ("active_concat", "concat"), ("active_concat", "noage_concat"),
                     ("cap", "mlp"), ("cap", "cagids"), ("cap", "noage"), ("cap_concat", "mlp"),
                     ("cap_concat", "concat"), ("cap_concat", "noage_concat"),
                     ("esx_cagids", "esx_mlp"), ("esx_concat", "esx_mlp"), ("esx_b3", "esx_mlp"),
                     ("esx_b4", "esx_mlp"), ("esx_nob", "esx_mlp"), ("esx_noage", "esx_mlp"),
                     ("esx_noage_concat", "esx_mlp"), ("esx_noage", "esx_cagids"),
                     ("esx_noage_concat", "esx_concat")]
    session_pairs += [("esx_" + k, k) for k in ("mlp", "b3", "b4", "concat", "cagids", "nob", "noage", "noage_concat")]
    for a, b in session_pairs:
        if SS.get(a) is None or SS.get(b) is None:
            continue
        res = two_level_bootstrap(SS[a], SS[b], ew_ss, n_classes, n_boot=args.n_boot)
        put("swv_{}_minus_{}".format(a, b), ci_text(res))
        put("swv_{}_minus_{}_excludes_zero".format(a, b), excludes_zero(res["all"]["two_level"]))
    # round 6: with validation on session-1 flows, gated CAG-IDS without C2 scores above the flow-only MLP on
    # session 2 (Table A7). Is that more than seed noise, and does it survive window-level validation?
    ssf = {k: seed_preds(pred_dir, "ss_" + t) for k, t in (("mlp", "flow_mlp"), ("nob", "abl_no_behavioural"))}
    if all(v is not None for v in ssf.values()):
        ew_ssf = np.load(work / "graphs_session" / "graphs.npz")["edge_window"]
        res = two_level_bootstrap(ssf["nob"], ssf["mlp"], ew_ssf, n_classes, n_boot=args.n_boot)
        put("ssf_nob_minus_mlp", ci_text(res))
        put("ssf_nob_minus_mlp_att", ci_text(res, "attacks"))
        put("ssf_nob_minus_mlp_excludes_zero", excludes_zero(res["all"]["two_level"]))
        ss_runs = {k: runs_of(results, "gnn_ss_" + t) for k, t in (("mlp", "flow_mlp"), ("nob", "abl_no_behavioural"))}
        gains = {c: np.mean([r["per_class"][c]["f1"] for r in ss_runs["nob"]]) -
                 np.mean([r["per_class"][c]["f1"] for r in ss_runs["mlp"]]) for c in classes}
        put("ssf_nob_minus_mlp_per_class", {c: signed(v) for c, v in gains.items()})
        for k in ("mlp", "nob"):
            put("ssf_{}_best_epoch".format(k), "{} to {}".format(min(r["best_epoch"] for r in ss_runs[k]),
                                                                 max(r["best_epoch"] for r in ss_runs[k])))
        if SS.get("nob") is not None:
            try:
                res = two_level_bootstrap(ssf["nob"], SS["nob"], ew_ssf, n_classes, n_boot=args.n_boot)
                put("ssf_nob_minus_swv_nob", ci_text(res))
            except ValueError as err:
                print("flow- and window-level session runs not comparable:", err)
    ss_rows = []
    # round 3: columns are validation and test under flow-level validation, the same under window-level
    # validation, and the test score when early stopping leaves out Backdoor and Worms
    for label, k in [("Flow-only MLP (B2)", "mlp"), ("E-GraphSAGE-style GNN (B3)", "b3"),
                     ("Centrality-initialised GNN (B4)", "b4"), ("CAG-IDS, concatenation", "concat"),
                     ("CAG-IDS, gated", "cagids"), ("CAG-IDS, gated, without C2", "nob"),
                     ("CAG-IDS, gated, without age", "noage"), ("CAG-IDS, gated, age clipped", "clip"),
                     ("CAG-IDS, gated, age over active windows", "active"),
                     ("CAG-IDS, gated, age capped at one hour", "cap"),
                     ("CAG-IDS, concatenation, without age", "noage_concat"),
                     ("CAG-IDS, concatenation, age clipped", "clip_concat"),
                     ("CAG-IDS, concatenation, age over active windows", "active_concat"),
                     ("CAG-IDS, concatenation, age capped at one hour", "cap_concat")]:
        if "swv_{}_macro".format(k) not in V:
            continue
        ss_rows.append([label, V.get("ssf_{}_val".format(k), "–"), V.get("ssf_{}_macro".format(k), "–"),
                        V["swv_{}_val".format(k)], V["swv_{}_macro".format(k)], V.get("swv_esx_{}_macro".format(k), "–")])
    V["table_session"] = ss_rows
    # the age feature counts intervals since a host was first seen, the gap between the sessions included
    sg = np.load(work / "graphs_session_wv" / "graphs.npz")
    s_ctx = np.load(work / "graphs_session_wv" / "context_raw.npy")
    s_split = np.load(work / "graphs_session_wv" / "edge_split.npy")
    s_names = json.load(open(work / "graphs_session_wv" / "meta.json", encoding="utf-8"))["context_names"]
    s_code = s_split[sg["edge_offsets"][:-1]]
    node_code = np.repeat(s_code, np.diff(sg["node_offsets"]))
    age = s_ctx[:, s_names.index("log_age_windows")]
    put("ss_age_train_range", "{:.1f} to {:.1f}".format(*np.percentile(age[node_code == 0], [1, 99])))
    put("ss_age_train_max", "{:.2f}".format(age[node_code == 0].max()))
    put("ss_age_test_range", "{:.1f} to {:.1f}".format(*np.percentile(age[node_code == 2], [1, 99])))
    put("ss_age_test_above_max", "{:.1f}%".format(100 * (age[node_code == 2] > age[node_code == 0].max()).mean()))
    # the attackers are active only in the first two hours of session 1, so in training their age stays small
    hosts = pd.read_parquet(work / "flows.parquet", columns=["src_ip", "dst_ip"])
    s_rows = np.load(work / "graphs_session_wv" / "row_index.npy")
    src = hosts["src_ip"].to_numpy(object)[s_rows]
    dst = hosts["dst_ip"].to_numpy(object)[s_rows]
    del hosts
    attackers = {"175.45.176.0", "175.45.176.1", "175.45.176.2", "175.45.176.3"}
    is_att = np.zeros(len(age), dtype=bool)
    eo_s, no_s = sg["edge_offsets"], sg["node_offsets"]
    for w in range(len(s_code)):
        nodes = np.unique(np.concatenate([src[eo_s[w]:eo_s[w + 1]], dst[eo_s[w]:eo_s[w + 1]]]))
        is_att[no_s[w]:no_s[w + 1]] = [n in attackers for n in nodes]
    put("ss_age_attackers_train_max", "{:.1f}".format(age[(node_code == 0) & is_att].max()))
    put("ss_age_others_train_max", "{:.1f}".format(age[(node_code == 0) & ~is_att].max()))
    # round 3: the same for the age over active windows and the age capped at one hour. These graphs hold the
    # same windows and hosts in the same order, so the node codes and attacker flags carry over.
    for tag, key in (("graphs_session_wv_activeage", "active"), ("graphs_session_wv_agecap", "cap")):
        if not (work / tag / "context_raw.npy").exists():
            continue
        a_names = json.load(open(work / tag / "meta.json", encoding="utf-8"))["context_names"]
        a_age = np.load(work / tag / "context_raw.npy")[:, a_names.index("log_age_windows")]
        if len(a_age) != len(age):
            raise ValueError("node order differs in " + tag)
        tr_max = a_age[node_code == 0].max()
        put("ss_age_{}_train_range".format(key), "{:.1f} to {:.1f}".format(*np.percentile(a_age[node_code == 0], [1, 99])))
        put("ss_age_{}_test_range".format(key), "{:.1f} to {:.1f}".format(*np.percentile(a_age[node_code == 2], [1, 99])))
        put("ss_age_{}_train_max".format(key), "{:.2f}".format(tr_max))
        put("ss_age_{}_test_above_max".format(key), "{:.1f}%".format(100 * (a_age[node_code == 2] > tr_max).mean()))
        put("ss_age_{}_attackers_train_max".format(key), "{:.1f}".format(a_age[(node_code == 0) & is_att].max()))
        put("ss_age_{}_others_train_max".format(key), "{:.1f}".format(a_age[(node_code == 0) & ~is_att].max()))
        put("ss_age_{}_attackers_test_range".format(key), "{:.1f} to {:.1f}".format(
            *np.percentile(a_age[(node_code == 2) & is_att], [1, 99])))
        put("ss_age_{}_attackers_test_above_attacker_max".format(key), "{:.1f}%".format(
            100 * (a_age[(node_code == 2) & is_att] > a_age[(node_code == 0) & is_att].max()).mean()))

    # ------------------------------------------------------------------ 7
    print("\n=== 7. smaller checks; Table 12 over 20 seeds ===")
    for name, base, alt in [("k1_b3", "egraphsage", "k1_egraphsage"), ("k1_cagids", "cagids", "k1_cagids"),
                            ("max_b3", "egraphsage", "max_egraphsage"), ("max_cagids", "cagids", "max_cagids"),
                            ("dir_b3", "egraphsage", "dir_egraphsage"), ("dir_cagids", "cagids", "dir_cagids")]:
        a, b = runs_of(results, "gnn_" + base), runs_of(results, "gnn_" + alt)
        if a and b:
            put("{}_macro".format(name), pm(macro_list(b)))
            put("{}_delta".format(name), signed(np.mean(macro_list(b)) - np.mean(macro_list(a))))
            res = two_level_bootstrap(seed_preds(pred_dir, alt), seed_preds(pred_dir, base), edge_window, n_classes,
                                      n_boot=args.n_boot)
            put("{}_ci".format(name), ci_text(res))

    abl12 = [("full", "Full context (C1 + C2 + C3)", "cagids"), ("no_c1", "without C1 (structural)", "abl_no_structural"),
             ("no_c2", "without C2 (behavioural)", "abl_no_behavioural"), ("no_c3", "without C3 (role)", "abl_no_role"),
             ("c1_only", "C1 only (C1 + gate control)", "c1_gate"), ("c4", "with C4 (time of day)", "abl_with_time")]
    rows12 = []
    for key, label, tag in abl12:
        # round 3: every row uses all of its seeds (20 for the full context, without C2 and C1 only)
        r = runs_all(results, tag)
        low = [np.mean([x["per_class"][c]["f1"] for c in LOW_VOLUME]) for x in r]
        low5 = [np.mean([x["per_class"][c]["f1"] for c in LOW_VOLUME + ["DoS"]]) for x in r]
        V["_low_" + key], V["_low5_" + key] = low, low5
        put("t12_{}_n".format(key), len(r))
        put("t12_{}_macro".format(key), pm(macro_list(r)))
        put("t12_{}_low".format(key), pm(low))
        put("t12_{}_low5".format(key), pm(low5))
        put("t12_{}_recon".format(key), f3(np.mean([x["per_class"]["Reconnaissance"]["f1"] for x in r])))
        rows12.append([label, str(len(r)), pm(macro_list(r)), pm(macro_list(r, "macro_f1_attacks"))] +
                      [f3(np.mean([x["per_class"][c]["f1"] for x in r])) for c in LOW_VOLUME] +
                      [pm(low), f3(np.mean([x["per_class"]["Reconnaissance"]["f1"] for x in r]))])
    V["table_ablation"] = rows12
    for k in ("no_c1", "no_c2", "no_c3"):
        put("t12_drop_{}".format(k), signed(np.mean(V["_low_" + k]) - np.mean(V["_low_full"])))
        put("t12_drop5_{}".format(k), signed(np.mean(V["_low5_" + k]) - np.mean(V["_low5_full"])))
        d, lo, hi = welch_interval(V["_low_" + k], V["_low_full"])
        put("t12_drop_{}_ci".format(k), "{} [{}, {}]".format(signed(d), signed(lo), signed(hi)))
        if k == "no_c2":
            # the largest drop that the interval cannot exclude, relative to the full-context mean
            put("t12_no_c2_max_drop", f3(-lo))
            put("t12_no_c2_max_drop_pct", "{:.0f}%".format(100 * -lo / np.mean(V["_low_full"])))
            dd5, lo5, hi5 = welch_interval(V["_low5_" + k], V["_low5_full"])
            put("t12_drop5_no_c2_ci", "{} [{}, {}]".format(signed(dd5), signed(lo5), signed(hi5)))
            hl = hodges_lehmann(V["_low_" + k], V["_low_full"])
            put("t12_drop_no_c2_hl", "{} [{}, {}]".format(signed(hl["hl"]), signed(hl["lo"]), signed(hl["hi"])))
            recon = {kk: [x["per_class"]["Reconnaissance"]["f1"] for x in runs_all(results, t)]
                     for kk, _, t in abl12 if kk in ("full", "no_c2")}
            dr, lr, hr = welch_interval(recon["no_c2"], recon["full"])
            put("t12_recon_no_c2_ci", "{} [{}, {}]".format(signed(dr), signed(lr), signed(hr)))
    drops = {k: np.mean(V["_low_full"]) - np.mean(V["_low_" + k]) for k in ("no_c1", "no_c2", "no_c3")}
    put("h2_c2_largest", "yes" if drops["no_c2"] > max(drops["no_c1"], drops["no_c3"]) else "no")
    put("h2_c2_beyond_sd", "yes" if drops["no_c2"] > max(np.std(V["_low_full"], ddof=1),
                                                         np.std(V["_low_no_c2"], ddof=1)) else "no")
    put("h2_supported", "yes" if (drops["no_c2"] > max(drops["no_c1"], drops["no_c3"]) and
                                  drops["no_c2"] > max(np.std(V["_low_full"], ddof=1),
                                                       np.std(V["_low_no_c2"], ddof=1))) else "no")
    drops5 = {k: np.mean(V["_low5_full"]) - np.mean(V["_low5_" + k]) for k in ("no_c1", "no_c2", "no_c3")}
    put("h2dos_supported", "yes" if (drops5["no_c2"] > max(drops5["no_c1"], drops5["no_c3"]) and
                                     drops5["no_c2"] > max(np.std(V["_low5_full"], ddof=1),
                                                           np.std(V["_low5_no_c2"], ddof=1))) else "no")
    macros12 = {k: np.mean(macro_list(runs_all(results, t))) for k, _, t in abl12}
    put("t12_macro_min", f3(min(macros12.values())))
    put("t12_macro_min_row", min(macros12, key=macros12.get))
    put("t12_macro_max", f3(max(macros12.values())))
    counts = meta["class_counts"]
    tot = {c: counts["train"][c] + counts["val"][c] + counts["test"][c] for c in classes}
    att_total = sum(v for c, v in tot.items() if c != "Benign")
    put("share_dos", "{:.1f}%".format(100 * tot["DoS"] / att_total))

    # the share of each random-flow gap that survives on the shared flows
    shares = {}
    for _, k, *_ in model_def:
        gap, same = V["_gap_rf_" + k], V["_same_rf_" + k]
        # round 3: the ratio of unrounded means (the convention of the text) and that of the rounded entries of
        # Tables 13 and 14, whose gap is the difference of the two rounded means
        table_gap = float(V["sp_{}_rf".format(k)].split(" ± ")[0]) - float(V["sp_{}_chrono".format(k)].split(" ± ")[0])
        table_same = float(V["sp_{}_same_rf".format(k)].split(" ")[0].replace("−", "-"))
        shares[k] = (same / gap, table_same / table_gap)
        put("sp_{}_gap_rf_table".format(k), signed(table_gap))
        put("share_{}".format(k), "{:.1f}% (rounded table entries {:.1f}%)".format(100 * same / gap, 100 * shares[k][1]))
    put("share_min", "{:.0f}%".format(100 * min(v[0] for v in shares.values())))
    put("share_max", "{:.0f}%".format(100 * max(v[0] for v in shares.values())))
    put("share_table_min", "{:.0f}%".format(100 * min(v[1] for v in shares.values())))
    put("share_table_max", "{:.0f}%".format(100 * max(v[1] for v in shares.values())))
    sw = runs_of(results, "gnn_sw_flow_mlp")
    if sw:
        put("sw_mlp_macro", pm(macro_list(sw)))
        put("sw_mlp_gap_rf", f3(np.mean(macro_list(runs_of(results, "baseline_random_forest"))) - np.mean(macro_list(sw))))

    rfr = load_json(results, "baseline_random_forest_random_flow")["aggregate"]["per_class"]["Backdoor"]["f1_mean"]
    ct = load_json(results, "context_probe_tests")["random_flow/flow_only"]["per_class_f1"]["Backdoor"]
    put("backdoor_random_baseline_5seeds", f3(rfr))
    put("backdoor_random_probe_3seeds", f3(ct))

    # neural training runs: 210 planned and post-hoc runs plus the robustness checks
    from run_all import EXTENDED_GNN_RUNS
    n_ext = sum(len(r[4]) for r in EXTENDED_GNN_RUNS if (results / "gnn_{}.json".format(r[0])).exists())
    put("n_extended_runs", n_ext)
    put("n_neural_runs", "{:,}".format(210 + n_ext))
    # the session split with whole-window validation and the extra ablation and random-split seeds
    newer = ("swv_", "extra_swv_", "extra_abl_", "extra_rf_")
    put("n_new_runs_round2", sum(len(r[4]) for r in EXTENDED_GNN_RUNS
                                 if r[0].startswith(newer) and (results / "gnn_{}.json".format(r[0])).exists()))
    round3 = ("noage_", "extra_noage_", "rf_noage_", "extra_rf_noage_", "swv_esx_", "swv_activeage_", "swv_agecap_",
              "extra_rf_flow_mlp", "extra_rf_egraphsage", "extra_rf_centrality_egraphsage", "extra_rf_cagids",
              "extra_abl_no_behavioural", "extra_rf_abl_no_behavioural")
    put("n_new_runs_round3", sum(len(r[4]) for r in EXTENDED_GNN_RUNS
                                 if r[0].startswith(round3) and (results / "gnn_{}.json".format(r[0])).exists()))
    put("n_ablation_extra", sum(1 for t, _ in [("abl_no_structural", 0), ("abl_no_behavioural", 0), ("abl_no_role", 0),
                                               ("abl_with_time", 0), ("rf_cagids", 0), ("rf_abl_no_structural", 0),
                                               ("rf_abl_no_behavioural", 0), ("rf_abl_no_role", 0), ("rf_c1_gate", 0)]
                                if (results / "gnn_extra_{}.json".format(t)).exists()))

    out = {k: v for k, v in V.items() if not k.startswith("_")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nwrote", args.out, "with", len(out), "values")


if __name__ == "__main__":
    main()
