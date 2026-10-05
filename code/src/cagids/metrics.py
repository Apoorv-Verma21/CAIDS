import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def false_positive_rate(y_true_binary, y_pred_binary):
    matrix = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp = matrix[0, 0], matrix[0, 1]
    denominator = tn + fp
    return float(fp) / float(denominator) if denominator else 0.0


def compute_metrics(y_true, y_pred, class_names, y_score=None, benign_index=0):
    results = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
    }

    true_binary = (np.asarray(y_true) != benign_index).astype(int)
    pred_binary = (np.asarray(y_pred) != benign_index).astype(int)
    results["binary_f1"] = float(f1_score(true_binary, pred_binary, zero_division=0))
    results["false_positive_rate"] = false_positive_rate(true_binary, pred_binary)

    if y_score is not None:
        score = np.asarray(y_score)
        if score.ndim == 2:
            attack_score = 1.0 - score[:, benign_index]
        else:
            attack_score = score
        if len(np.unique(true_binary)) > 1:
            results["pr_auc"] = float(average_precision_score(true_binary, attack_score))
        else:
            results["pr_auc"] = float("nan")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    results["per_class"] = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    return results


def format_metrics(name, results):
    lines = ["", "=" * 62, name, "=" * 62]
    lines.append("macro F1        : {:.4f}".format(results["macro_f1"]))
    lines.append("weighted F1     : {:.4f}".format(results["weighted_f1"]))
    lines.append("binary F1       : {:.4f}".format(results["binary_f1"]))
    lines.append("false pos. rate : {:.4f}".format(results["false_positive_rate"]))
    if "pr_auc" in results:
        lines.append("PR-AUC (binary) : {:.4f}".format(results["pr_auc"]))
    lines.append("")
    lines.append("{:<20} {:>9} {:>9} {:>9} {:>9}".format("class", "prec", "recall", "f1", "support"))
    for cls, values in results["per_class"].items():
        lines.append("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}".format(
            cls[:20], values["precision"], values["recall"], values["f1"], values["support"]
        ))
    return "\n".join(lines)


def aggregate_seeds(runs):
    if not runs:
        return {}
    keys = ["macro_f1", "weighted_f1", "binary_f1", "false_positive_rate", "pr_auc"]
    out = {}
    for key in keys:
        values = [r[key] for r in runs if key in r and not np.isnan(r.get(key, np.nan))]
        if values:
            out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    out["n_seeds"] = len(runs)
    return out
