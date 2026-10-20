import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def false_positive_rate(y_true_binary, y_pred_binary):
    matrix = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp = matrix[0, 0], matrix[0, 1]
    denominator = tn + fp
    return float(fp) / float(denominator) if denominator else 0.0


def present_labels(y_true, n_classes):
    # Macro averages are taken over the classes that actually occur in the
    # evaluated split. A class with no support cannot be scored, and letting
    # it in only when a model happens to predict it would make the macro F1 of
    # two models average over different sets of classes.
    counts = np.bincount(np.asarray(y_true, dtype="int64"), minlength=n_classes)
    return [i for i in range(n_classes) if counts[i] > 0]


def macro_f1_present(y_true, y_pred, n_classes, keep=None):
    """Macro-F1 over the classes present in y_true, optionally only over the
    class indices in keep."""
    labels = present_labels(y_true, n_classes)
    if keep is not None:
        labels = [i for i in labels if i in set(keep)]
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def compute_metrics(y_true, y_pred, class_names, y_score=None, benign_index=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names)
    labels = present_labels(y_true, n_classes)

    results = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "classes_scored": [class_names[i] for i in labels],
    }

    attack_labels = [i for i in labels if i != benign_index]
    if attack_labels:
        results["macro_f1_attacks"] = float(
            f1_score(y_true, y_pred, labels=attack_labels, average="macro", zero_division=0)
        )

    true_binary = (y_true != benign_index).astype(int)
    pred_binary = (y_pred != benign_index).astype(int)
    results["binary_f1"] = float(f1_score(true_binary, pred_binary, zero_division=0))
    b_prec, b_rec, _, _ = precision_recall_fscore_support(
        true_binary, pred_binary, labels=[1], zero_division=0
    )
    results["binary_precision"] = float(b_prec[0])
    results["detection_rate"] = float(b_rec[0])
    results["false_positive_rate"] = false_positive_rate(true_binary, pred_binary)

    if y_score is not None:
        score = np.asarray(y_score, dtype="float64")
        attack_score = 1.0 - score[:, benign_index] if score.ndim == 2 else score
        if len(np.unique(true_binary)) > 1:
            results["pr_auc"] = float(average_precision_score(true_binary, attack_score))
            results["roc_auc"] = float(roc_auc_score(true_binary, attack_score))
        else:
            results["pr_auc"] = float("nan")
            results["roc_auc"] = float("nan")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    results["per_class"] = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }
    results["confusion"] = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    return results


def format_metrics(name, results):
    lines = ["", "=" * 62, name, "=" * 62]
    lines.append("macro F1        : {:.4f}  (over {} classes present)".format(
        results["macro_f1"], len(results.get("classes_scored", []))))
    if "macro_f1_attacks" in results:
        lines.append("macro F1 attacks: {:.4f}".format(results["macro_f1_attacks"]))
    lines.append("weighted F1     : {:.4f}".format(results["weighted_f1"]))
    lines.append("binary F1       : {:.4f}".format(results["binary_f1"]))
    lines.append("detection rate  : {:.4f}".format(results["detection_rate"]))
    lines.append("false pos. rate : {:.5f}".format(results["false_positive_rate"]))
    if "pr_auc" in results:
        lines.append("PR-AUC (binary) : {:.4f}".format(results["pr_auc"]))
    lines.append("")
    lines.append("{:<20} {:>9} {:>9} {:>9} {:>9}".format("class", "prec", "recall", "f1", "support"))
    for cls, values in results["per_class"].items():
        lines.append("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}".format(
            cls[:20], values["precision"], values["recall"], values["f1"], values["support"]
        ))
    return "\n".join(lines)


AGGREGATE_KEYS = [
    "macro_f1", "macro_f1_attacks", "weighted_f1", "binary_f1", "binary_precision",
    "detection_rate", "false_positive_rate", "pr_auc", "roc_auc", "accuracy",
]


def aggregate_seeds(runs):
    if not runs:
        return {}
    out = {}
    for key in AGGREGATE_KEYS:
        values = [r[key] for r in runs if key in r and not np.isnan(r.get(key, np.nan))]
        if values:
            out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    classes = list(runs[0].get("per_class", {}).keys())
    per_class = {}
    for cls in classes:
        f1s = [r["per_class"][cls]["f1"] for r in runs]
        per_class[cls] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "support": runs[0]["per_class"][cls]["support"],
        }
    out["per_class"] = per_class
    out["n_seeds"] = len(runs)
    return out
