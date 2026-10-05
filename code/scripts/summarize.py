import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cagids.config import load_config

HEADERS = ["model", "macro F1", "weighted F1", "binary F1", "FPR", "PR-AUC"]


def collect(results_dir):
    rows = []

    baseline_path = results_dir / "baselines.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for name, metrics in payload.get("results", {}).items():
            rows.append({
                "model": name,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "binary_f1": metrics["binary_f1"],
                "fpr": metrics["false_positive_rate"],
                "pr_auc": metrics.get("pr_auc", float("nan")),
                "std": None,
                "per_class": metrics["per_class"],
            })

    for path in sorted(results_dir.glob("gnn_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        runs = payload.get("runs", [])
        if not runs:
            continue
        agg = payload.get("aggregate", {})
        first = runs[0]
        rows.append({
            "model": payload.get("experiment", path.stem),
            "macro_f1": agg.get("macro_f1", {}).get("mean", first["macro_f1"]),
            "weighted_f1": agg.get("weighted_f1", {}).get("mean", first["weighted_f1"]),
            "binary_f1": agg.get("binary_f1", {}).get("mean", first["binary_f1"]),
            "fpr": agg.get("false_positive_rate", {}).get("mean", first["false_positive_rate"]),
            "pr_auc": agg.get("pr_auc", {}).get("mean", first.get("pr_auc", float("nan"))),
            "std": agg.get("macro_f1", {}).get("std"),
            "per_class": first["per_class"],
        })

    return rows


def as_text(rows):
    lines = ["{:<26} {:>10} {:>12} {:>10} {:>8} {:>8}".format(*HEADERS), "-" * 78]
    for row in rows:
        macro = "{:.4f}".format(row["macro_f1"])
        if row.get("std"):
            macro += " ±{:.3f}".format(row["std"])
        lines.append("{:<26} {:>10} {:>12.4f} {:>10.4f} {:>8.4f} {:>8.4f}".format(
            row["model"][:26], macro, row["weighted_f1"], row["binary_f1"],
            row["fpr"], row["pr_auc"],
        ))
    return "\n".join(lines)


def as_latex(rows):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Detection performance on the chronological test split.}",
        "\\label{tab:results}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Model & Macro F1 & Weighted F1 & Binary F1 & FPR & PR-AUC \\\\",
        "\\hline",
    ]
    for row in rows:
        macro = "{:.4f}".format(row["macro_f1"])
        if row.get("std"):
            macro = "{:.4f} $\\pm$ {:.3f}".format(row["macro_f1"], row["std"])
        lines.append("{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
            row["model"].replace("_", "\\_"), macro, row["weighted_f1"],
            row["binary_f1"], row["fpr"], row["pr_auc"],
        ))
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def per_class_table(rows):
    if not rows:
        return ""
    classes = list(rows[0]["per_class"].keys())
    header = "{:<26}".format("model") + "".join("{:>14}".format(c[:13]) for c in classes)
    lines = ["", "Per-class F1", header, "-" * len(header)]
    for row in rows:
        line = "{:<26}".format(row["model"][:26])
        for cls in classes:
            value = row["per_class"].get(cls, {}).get("f1", float("nan"))
            line += "{:>14.4f}".format(value)
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect all results into one comparison table")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    cfg = load_config(args.config, project_root=args.root)
    results_dir = cfg.get_path("results_dir")
    rows = collect(results_dir)

    if not rows:
        print("No results found in {}.".format(results_dir))
        print("Run run_baselines.py and train_gnn.py first.")
        return

    order = {"random_forest": 0, "xgboost": 1, "mlp": 2,
             "egraphsage": 3, "centrality_egraphsage": 4,
             "cagids_concat": 5, "cagids": 6}
    rows.sort(key=lambda r: order.get(r["model"], 99))

    print(as_text(rows))
    print(per_class_table(rows))

    out_path = results_dir / "summary.txt"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(as_text(rows) + "\n" + per_class_table(rows) + "\n")

    if args.latex:
        latex = as_latex(rows)
        print("\n" + latex)
        with open(results_dir / "results_table.tex", "w", encoding="utf-8") as handle:
            handle.write(latex + "\n")

    print("\nsaved {}".format(out_path))


if __name__ == "__main__":
    main()
