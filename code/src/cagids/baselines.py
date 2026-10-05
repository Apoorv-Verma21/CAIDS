import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .metrics import compute_metrics


def _subsample(x, y, limit, seed=42):
    if limit is None or len(y) <= limit:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=int(limit), replace=False)
    idx.sort()
    return x[idx], y[idx]


def run_random_forest(x_train, y_train, x_test, y_test, class_names, cfg):
    model = RandomForestClassifier(
        n_estimators=int(cfg.baselines.get("n_estimators", 200)),
        max_depth=cfg.baselines.get("max_depth"),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(cfg.train.get("seed", 42)),
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)
    return compute_metrics(y_test, probs.argmax(axis=1), class_names, probs)


def run_xgboost(x_train, y_train, x_test, y_test, class_names, cfg):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    counts = np.bincount(y_train, minlength=len(class_names)).astype("float64")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(class_names) * counts)
    sample_weight = weights[y_train]

    model = XGBClassifier(
        n_estimators=int(cfg.baselines.get("n_estimators", 200)),
        max_depth=6,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(class_names),
        tree_method="hist",
        n_jobs=-1,
        random_state=int(cfg.train.get("seed", 42)),
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    probs = model.predict_proba(x_test)
    return compute_metrics(y_test, probs.argmax(axis=1), class_names, probs)


def run_mlp(x_train, y_train, x_test, y_test, class_names, cfg):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=int(cfg.train.get("seed", 42)),
    )
    model.fit(x_train_scaled, y_train)
    probs = model.predict_proba(x_test_scaled)
    return compute_metrics(y_test, probs.argmax(axis=1), class_names, probs)


RUNNERS = {
    "random_forest": run_random_forest,
    "xgboost": run_xgboost,
    "mlp": run_mlp,
}


def run_baselines(x_train, y_train, x_test, y_test, class_names, cfg, verbose=True):
    limit = cfg.baselines.get("subsample_train")
    x_small, y_small = _subsample(x_train, y_train, limit, int(cfg.train.get("seed", 42)))

    results = {}
    for name in cfg.baselines.get("models", list(RUNNERS)):
        runner = RUNNERS.get(name)
        if runner is None:
            continue
        if verbose:
            print("running baseline: {}".format(name), flush=True)
        outcome = runner(x_small, y_small, x_test, y_test, class_names, cfg)
        if outcome is not None:
            results[name] = outcome
    return results
