import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .metrics import compute_metrics


def subsample_keep_attacks(x, y, limit, seed=42, benign_index=0):
    """Keep every attack flow and fill the rest of the budget with benign flows,
    so rare classes are never thinned out by the subsample."""
    if limit is None or len(y) <= limit:
        return x, y
    rng = np.random.default_rng(seed)
    attack_idx = np.flatnonzero(y != benign_index)
    benign_idx = np.flatnonzero(y == benign_index)
    n_benign = max(0, int(limit) - len(attack_idx))
    if n_benign < len(benign_idx):
        benign_idx = rng.choice(benign_idx, size=n_benign, replace=False)
    idx = np.sort(np.concatenate([attack_idx, benign_idx]))
    return x[idx], y[idx]


def _inverse_frequency(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype("float64")
    counts[counts == 0] = 1.0
    return counts.sum() / (n_classes * counts)


def run_random_forest(x_train, y_train, x_test, y_test, class_names, cfg, seed):
    model = RandomForestClassifier(
        n_estimators=int(cfg.baselines.get("n_estimators", 200)),
        max_depth=cfg.baselines.get("max_depth"),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    t0 = time.perf_counter()
    probs = _full_proba(model, model.predict_proba(x_test), len(class_names))
    infer = time.perf_counter() - t0
    return probs, infer


def run_xgboost(x_train, y_train, x_test, y_test, class_names, cfg, seed):
    from xgboost import XGBClassifier

    n_classes = len(class_names)
    # XGBoost needs contiguous labels 0..k-1, so absent classes are remapped.
    present = np.unique(y_train)
    remap = {c: i for i, c in enumerate(present)}
    y_local = np.array([remap[c] for c in y_train])
    sample_weight = _inverse_frequency(y_local, len(present))[y_local]

    model = XGBClassifier(
        n_estimators=int(cfg.baselines.get("n_estimators", 200)),
        max_depth=6,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_local, sample_weight=sample_weight, verbose=False)
    t0 = time.perf_counter()
    local = model.predict_proba(x_test)
    infer = time.perf_counter() - t0
    probs = np.zeros((len(x_test), n_classes), dtype="float64")
    probs[:, present] = local
    return probs, infer


def run_sklearn_mlp(x_train, y_train, x_test, y_test, class_names, cfg, seed):
    scaler = StandardScaler()
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=seed,
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    t0 = time.perf_counter()
    probs = _full_proba(model, model.predict_proba(scaler.transform(x_test)), len(class_names))
    infer = time.perf_counter() - t0
    return probs, infer


def _full_proba(model, probs, n_classes):
    full = np.zeros((probs.shape[0], n_classes), dtype="float64")
    full[:, model.classes_] = probs
    return full


RUNNERS = {
    "random_forest": run_random_forest,
    "xgboost": run_xgboost,
    "sklearn_mlp": run_sklearn_mlp,
}


def run_baseline(name, x_train, y_train, x_test, y_test, class_names, cfg, seed, verbose=True, x_val=None):
    """Returns the test metrics, the test probabilities and, when x_val is
    given, the validation probabilities of the same fitted model (else None)."""
    runner = RUNNERS[name]
    limit = cfg.baselines.get("subsample_train")
    x_small, y_small = subsample_keep_attacks(x_train, y_train, limit, seed)
    if verbose:
        print("running baseline: {} (seed {}, {} training flows)".format(name, seed, len(y_small)), flush=True)
    x_eval = x_test if x_val is None else np.concatenate([x_test, x_val])
    t0 = time.perf_counter()
    probs, infer_seconds = runner(x_small, y_small, x_eval, y_test, class_names, cfg, seed)
    total = time.perf_counter() - t0
    val_probs = None
    if x_val is not None:
        probs, val_probs = probs[:len(x_test)], probs[len(x_test):]
    metrics = compute_metrics(y_test, probs.argmax(axis=1), class_names, probs)
    metrics["train_seconds"] = total - infer_seconds
    metrics["inference"] = {"flows_per_second": float(len(x_eval) / max(infer_seconds, 1e-9))}
    metrics["n_train_flows"] = int(len(y_small))
    return metrics, probs, val_probs
