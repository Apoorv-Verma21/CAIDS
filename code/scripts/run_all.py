"""Run the full experiment suite in a fixed order. Finished steps are skipped,
so the script can be stopped and restarted.

    python scripts/run_all.py --root D:/CAIDS_work --config configs/default.yaml
    python scripts/run_all.py --root D:/CAIDS_work --config configs/default.yaml --extended
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")

SEEDS = ["1", "2", "3", "4", "5"]

GRAPH_BUILDS = [
    ("graphs", []),
    ("graphs_random_flow", ["--split", "random_flow"]),
    ("graphs_no_structural", ["--no-structural"]),
    ("graphs_no_behavioural", ["--no-behavioural"]),
    ("graphs_no_role", ["--no-role"]),
    ("graphs_with_time", ["--with-time"]),
    ("graphs_w30", ["--window-seconds", "30"]),
    ("graphs_w300", ["--window-seconds", "300"]),
    # C1 only: the gated model with structural context alone, which separates the
    # extra context groups from the direct context path (CAG-IDS against B4)
    ("graphs_c1_only", ["--no-behavioural", "--no-role"]),
]

# (results tag, experiment, graphs, extra args)
GNN_RUNS = [
    ("flow_mlp", "flow_mlp", "graphs", []),
    ("egraphsage", "egraphsage", "graphs", []),
    ("centrality_egraphsage", "centrality_egraphsage", "graphs", []),
    ("cagids_concat", "cagids_concat", "graphs", []),
    ("cagids", "cagids", "graphs", []),
    ("context_only", "context_only", "graphs", []),
    ("abl_no_structural", "cagids", "graphs_no_structural", []),
    ("abl_no_behavioural", "cagids", "graphs_no_behavioural", []),
    ("abl_no_role", "cagids", "graphs_no_role", []),
    ("abl_with_time", "cagids", "graphs_with_time", []),
    ("rf_flow_mlp", "flow_mlp", "graphs_random_flow", []),
    ("rf_egraphsage", "egraphsage", "graphs_random_flow", []),
    ("rf_centrality_egraphsage", "centrality_egraphsage", "graphs_random_flow", []),
    ("rf_cagids_concat", "cagids_concat", "graphs_random_flow", []),
    ("rf_cagids", "cagids", "graphs_random_flow", []),
    ("w30_cagids", "cagids", "graphs_w30", []),
    ("w300_cagids", "cagids", "graphs_w300", []),
    ("w30_egraphsage", "egraphsage", "graphs_w30", []),
    ("w300_egraphsage", "egraphsage", "graphs_w300", []),
    ("w30_flow_mlp", "flow_mlp", "graphs_w30", []),
    ("w300_flow_mlp", "flow_mlp", "graphs_w300", []),
    ("w30_centrality_egraphsage", "centrality_egraphsage", "graphs_w30", []),
    ("w300_centrality_egraphsage", "centrality_egraphsage", "graphs_w300", []),
    ("w30_cagids_concat", "cagids_concat", "graphs_w30", []),
    ("w300_cagids_concat", "cagids_concat", "graphs_w300", []),
    ("cagids_gatv2", "cagids_gatv2", "graphs", []),
    ("c1_gate", "cagids", "graphs_c1_only", []),
    # Post-hoc training sensitivity, added after the planned runs showed that
    # every neural model over-predicts the rare classes (high recall, very low
    # precision). These runs check whether the conclusions depend on the
    # class weighting, the loss or the training budget; the planned settings
    # stay the primary results.
    ("sw_flow_mlp", "flow_mlp", "graphs", ["--class-weighting", "sqrt_inverse"]),
    ("sw_egraphsage", "egraphsage", "graphs", ["--class-weighting", "sqrt_inverse"]),
    ("sw_centrality_egraphsage", "centrality_egraphsage", "graphs", ["--class-weighting", "sqrt_inverse"]),
    ("sw_cagids_concat", "cagids_concat", "graphs", ["--class-weighting", "sqrt_inverse"]),
    ("sw_cagids", "cagids", "graphs", ["--class-weighting", "sqrt_inverse"]),
    ("focal_flow_mlp", "flow_mlp", "graphs", ["--loss", "focal"]),
    ("focal_egraphsage", "egraphsage", "graphs", ["--loss", "focal"]),
    ("focal_centrality_egraphsage", "centrality_egraphsage", "graphs", ["--loss", "focal"]),
    ("focal_cagids_concat", "cagids_concat", "graphs", ["--loss", "focal"]),
    ("cagids_focal", "cagids", "graphs", ["--loss", "focal"]),
    ("long_flow_mlp", "flow_mlp", "graphs", ["--epochs", "60", "--patience", "12"]),
    ("long_egraphsage", "egraphsage", "graphs", ["--epochs", "60", "--patience", "12"]),
    ("long_centrality_egraphsage", "centrality_egraphsage", "graphs", ["--epochs", "60", "--patience", "12"]),
    ("long_cagids_concat", "cagids_concat", "graphs", ["--epochs", "60", "--patience", "12"]),
    ("long_cagids", "cagids", "graphs", ["--epochs", "60", "--patience", "12"]),
]

BASELINE_RUNS = [
    ("graphs", None),
    ("graphs_random_flow", "random_flow"),
]

# ----------------------------------------------------------------------------
# Robustness checks (run with --extended). They were added after the planned
# runs and do not change any of the planned runs above.

EXTRA_SEEDS = [str(s) for s in range(6, 21)]
FINGERPRINT_FIELDS = ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_QUERY_ID"]
FIVE = [("flow_mlp", "flow_mlp"), ("egraphsage", "egraphsage"), ("centrality_egraphsage", "centrality_egraphsage"),
        ("cagids_concat", "cagids_concat"), ("cagids", "cagids")]

# the context-group ablations under both splits (C4 on the chronological split only)
ABLATIONS = [("abl_no_structural", "graphs_no_structural"), ("abl_no_behavioural", "graphs_no_behavioural"),
             ("abl_no_role", "graphs_no_role"), ("abl_with_time", "graphs_with_time"),
             ("rf_abl_no_structural", "graphs_rf_no_structural"), ("rf_abl_no_behavioural", "graphs_rf_no_behavioural"),
             ("rf_abl_no_role", "graphs_rf_no_role"), ("rf_c1_gate", "graphs_rf_c1_only")]

# session split with whole session-1 windows held out for validation, and gated and
# concatenated CAG-IDS without the age feature or with it clipped at its training maximum
AGE = "log_age_windows"
RAREST = ["Backdoor", "Worms"]
SESSION_WV = ([("swv_" + tag, exp, "graphs_session_wv", []) for tag, exp in FIVE] +
              [("swv_abl_no_behavioural", "cagids", "graphs_session_wv_no_behavioural", [])] +
              [("swv_noage_" + tag, exp, "graphs_session_wv", ["--drop-context", AGE])
               for tag, exp in FIVE if tag.startswith("cagids")] +
              [("swv_clipage_" + tag, exp, "graphs_session_wv", ["--clip-context", AGE])
               for tag, exp in FIVE if tag.startswith("cagids")])

EXTENDED_GRAPH_BUILDS = [
    # whole windows assigned at random: no shared windows, but train and test interleaved in time
    ("graphs_random_window", ["--split", "random"]),
    # train on capture session 1, test on session 2
    ("graphs_session", ["--split", "session"]),
    # context-group ablations under the random flow split
    ("graphs_rf_no_structural", ["--split", "random_flow", "--no-structural"]),
    ("graphs_rf_no_behavioural", ["--split", "random_flow", "--no-behavioural"]),
    ("graphs_rf_no_role", ["--split", "random_flow", "--no-role"]),
    ("graphs_rf_c1_only", ["--split", "random_flow", "--no-behavioural", "--no-role"]),
    # 30-s and 300-s windows with the training, validation and test periods of the 60-s split
    ("graphs_w30_aligned", ["--window-seconds", "30", "--split", "aligned"]),
    ("graphs_w300_aligned", ["--window-seconds", "300", "--split", "aligned"]),
    # session split without the behavioural group, whose age feature leaves the training range in session 2
    ("graphs_session_no_behavioural", ["--split", "session", "--no-behavioural"]),
    # session split validated on whole session-1 windows, with and without the behavioural group
    ("graphs_session_wv", ["--split", "session", "--session-val", "windows"]),
    ("graphs_session_wv_no_behavioural", ["--split", "session", "--session-val", "windows", "--no-behavioural"]),
    # the same with an age that counts only the windows in which a host was active, or that stops at one hour
    ("graphs_session_wv_activeage", ["--split", "session", "--session-val", "windows", "--age-mode", "active"]),
    ("graphs_session_wv_agecap", ["--split", "session", "--session-val", "windows", "--age-cap", "60"]),
]

# (results tag, experiment, graphs, extra args, seeds)
EXTENDED_GNN_RUNS = (
    # seeds 6 to 20 for the models behind H1, H3 and the C1 + gate control
    [("extra_" + tag, exp, "graphs", [], EXTRA_SEEDS) for tag, exp in FIVE] +
    [("extra_c1_gate", "cagids", "graphs_c1_only", [], EXTRA_SEEDS),
     ("extra_cagids_gatv2", "cagids_gatv2", "graphs", [], EXTRA_SEEDS)] +
    # the forest's training set: all attack flows plus benign flows up to 400,000, weights from that sample
    [("sub_" + tag, exp, "graphs", ["--train-subsample", "400000"], SEEDS)
     for tag, exp in FIVE if tag in ("flow_mlp", "centrality_egraphsage", "cagids")] +
    # gated CAG-IDS with one context group removed, random flow split
    [("rf_abl_no_structural", "cagids", "graphs_rf_no_structural", [], SEEDS),
     ("rf_abl_no_behavioural", "cagids", "graphs_rf_no_behavioural", [], SEEDS),
     ("rf_abl_no_role", "cagids", "graphs_rf_no_role", [], SEEDS),
     ("rf_c1_gate", "cagids", "graphs_rf_c1_only", [], SEEDS)] +
    [("rw_" + tag, exp, "graphs_random_window", [], SEEDS) for tag, exp in FIVE] +
    [("ss_" + tag, exp, "graphs_session", [], SEEDS) for tag, exp in FIVE] +
    [("ss_abl_no_behavioural", "cagids", "graphs_session_no_behavioural", [], SEEDS)] +
    # possible machine fingerprints removed from the flow vector
    [("nofp_flow_mlp", "flow_mlp", "graphs", ["--drop-features"] + FINGERPRINT_FIELDS, SEEDS),
     ("nofp_cagids", "cagids", "graphs", ["--drop-features"] + FINGERPRINT_FIELDS, SEEDS)] +
    # one message-passing layer, max aggregation, direction-aware messages
    [("k1_egraphsage", "egraphsage", "graphs", ["--layers", "1"], SEEDS),
     ("k1_cagids", "cagids", "graphs", ["--layers", "1"], SEEDS),
     ("max_egraphsage", "egraphsage", "graphs", ["--aggregator", "max"], SEEDS),
     ("max_cagids", "cagids", "graphs", ["--aggregator", "max"], SEEDS),
     ("dir_egraphsage", "egraphsage", "graphs", ["--direction-aware"], SEEDS),
     ("dir_cagids", "cagids", "graphs", ["--direction-aware"], SEEDS)] +
    # window length with mini-batches of about 13,500 flows and the 60-s periods
    [("wa30_" + tag, exp, "graphs_w30_aligned", ["--windows-per-batch", "16"], SEEDS) for tag, exp in FIVE] +
    [("wa300_" + tag, exp, "graphs_w300_aligned", ["--windows-per-batch", "2"], SEEDS) for tag, exp in FIVE] +
    # seeds 6 to 20 for the context-group ablations and for the five neural models under both random splits
    [("extra_" + tag, "cagids", graphs, [], EXTRA_SEEDS) for tag, graphs in ABLATIONS] +
    [("extra_rf_" + tag, exp, "graphs_random_flow", [], EXTRA_SEEDS) for tag, exp in FIVE] +
    [("extra_rw_" + tag, exp, "graphs_random_window", [], EXTRA_SEEDS) for tag, exp in FIVE] +
    # the session split with whole-window validation, seeds 1 to 20
    [(tag, exp, graphs, extra, SEEDS) for tag, exp, graphs, extra in SESSION_WV] +
    [("extra_" + tag, exp, graphs, extra, EXTRA_SEEDS) for tag, exp, graphs, extra in SESSION_WV] +
    # gated CAG-IDS without the age feature on the chronological and the random flow split, seeds 1 to 20:
    # does age carry the random-split gain of the neural model?
    [("noage_cagids", "cagids", "graphs", ["--drop-context", AGE], SEEDS),
     ("extra_noage_cagids", "cagids", "graphs", ["--drop-context", AGE], EXTRA_SEEDS),
     ("rf_noage_cagids", "cagids", "graphs_random_flow", ["--drop-context", AGE], SEEDS),
     ("extra_rf_noage_cagids", "cagids", "graphs_random_flow", ["--drop-context", AGE], EXTRA_SEEDS)] +
    # session split: early stopping on the macro-F1 of the eight classes other than Backdoor and Worms, of
    # which the validation windows hold one and two flows
    [("swv_esx_" + tag, exp, "graphs_session_wv", ["--early-stop-exclude"] + RAREST, SEEDS) for tag, exp in FIVE] +
    [("swv_esx_abl_no_behavioural", "cagids", "graphs_session_wv_no_behavioural", ["--early-stop-exclude"] + RAREST,
      SEEDS)] +
    [("swv_esx_noage_" + tag, exp, "graphs_session_wv", ["--drop-context", AGE, "--early-stop-exclude"] + RAREST,
      SEEDS) for tag, exp in FIVE if tag.startswith("cagids")] +
    # session split with an age that a gap in the capture cannot move: counted over active windows only,
    # or capped at one hour
    [("swv_activeage_" + tag, exp, "graphs_session_wv_activeage", [], SEEDS) for tag, exp in FIVE
     if tag.startswith("cagids")] +
    [("swv_agecap_" + tag, exp, "graphs_session_wv_agecap", [], SEEDS) for tag, exp in FIVE
     if tag.startswith("cagids")] +
    # the seeds 6 to 20 of B3 and B4 that label the victims' replies as attacks under the planned settings,
    # retrained with the longer budget: does the failure go away when the same seed trains longer?
    [("failrerun_egraphsage", "egraphsage", "graphs", ["--epochs", "60", "--patience", "12"],
      ["8", "10", "11", "13", "19"]),
     ("failrerun_centrality_egraphsage", "centrality_egraphsage", "graphs", ["--epochs", "60", "--patience", "12"],
      ["8", "11", "13", "14", "15", "16", "19"])]
)

# (graphs, results tag, extra args)
EXTENDED_BASELINE_RUNS = [
    ("graphs", "withval", ["--models", "random_forest", "--save-val"]),
    ("graphs", "nofp", ["--models", "random_forest", "--drop-features"] + FINGERPRINT_FIELDS),
    ("graphs_random_window", "random_window", []),
    ("graphs_session", "session", []),
    ("graphs_session_wv", "session_wv", []),
]


def run(cmd, log):
    line = " ".join(cmd)
    print("\n>>> " + line, flush=True)
    log.write("\n>>> " + line + "\n")
    log.flush()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write("<<< exit {} after {:.0f}s\n".format(proc.returncode, time.perf_counter() - t0))
    log.flush()
    print("<<< exit {} after {:.0f}s".format(proc.returncode, time.perf_counter() - t0), flush=True)
    if proc.returncode != 0:
        raise SystemExit("step failed: " + line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="run only these GNN tags")
    parser.add_argument("--skip-builds", action="store_true")
    parser.add_argument("--extended", "--revision", dest="extended", action="store_true",
                        help="run the robustness checks instead of the planned runs")
    parser.add_argument("--builds-only", action="store_true", help="with --extended: build the graphs and stop")
    parser.add_argument("--only-build", nargs="*", default=None, help="with --extended: build only these graph tags")
    parser.add_argument("--baselines-only", action="store_true", help="with --extended: run the flat baselines and stop")
    parser.add_argument("--log", default="run_all.log", help="log file inside the results folder")
    args = parser.parse_args()

    py = sys.executable
    root = Path(args.root)
    work = root / "outputs" / "work"
    results = root / "outputs" / "results"
    results.mkdir(parents=True, exist_ok=True)
    common = ["--config", args.config, "--root", str(root)]
    log = open(results / args.log, "a", encoding="utf-8")

    if args.extended:
        run_extended(args, py, common, work, results, log)
        return

    if not (work / "flows.parquet").exists():
        run([py, os.path.join(SCRIPTS, "prepare_data.py")] + common, log)

    if not args.skip_builds:
        for tag, flags in GRAPH_BUILDS:
            if not (work / tag / "edge_split.npy").exists():
                run([py, os.path.join(SCRIPTS, "build_graphs.py")] + common + flags + ["--tag", tag], log)

    if not (results / "diagnostics_graphs.json").exists():
        run([py, os.path.join(SCRIPTS, "diagnostics.py")] + common + ["--graphs", "graphs"], log)

    for graphs, tag in BASELINE_RUNS:
        suffix = "" if tag is None else "_" + tag
        if not (results / "baseline_xgboost{}.json".format(suffix)).exists():
            cmd = [py, os.path.join(SCRIPTS, "run_baselines.py")] + common + ["--graphs", graphs, "--seeds"] + SEEDS
            if tag:
                cmd += ["--tag", tag]
            run(cmd, log)

    for tag, experiment, graphs, extra in GNN_RUNS:
        if args.only and tag not in args.only:
            continue
        if (results / "gnn_{}.json".format(tag)).exists():
            continue
        if not (work / graphs / "edge_split.npy").exists():
            print("skipping {}: {} not built".format(tag, graphs))
            continue
        run([py, os.path.join(SCRIPTS, "train_gnn.py")] + common +
            ["--experiment", experiment, "--graphs", graphs, "--tag", tag, "--seeds"] + SEEDS + extra, log)

    if not args.only and not (results / "context_probe_tests.json").exists():
        run([py, os.path.join(SCRIPTS, "context_probe_tests.py")] + common, log)

    print("\nall steps finished", flush=True)


def run_extended(args, py, common, work, results, log):
    if not args.skip_builds:
        for tag, flags in EXTENDED_GRAPH_BUILDS:
            if args.only_build is not None and tag not in args.only_build:
                continue
            if not (work / tag / "edge_split.npy").exists():
                run([py, os.path.join(SCRIPTS, "build_graphs.py")] + common + flags + ["--tag", tag], log)
    if args.builds_only:
        return

    if args.only is None:
        for graphs, tag, extra in EXTENDED_BASELINE_RUNS:
            done = results / "baseline_random_forest_{}.json".format(tag)
            if done.exists() or not (work / graphs / "edge_split.npy").exists():
                continue
            run([py, os.path.join(SCRIPTS, "run_baselines.py")] + common +
                ["--graphs", graphs, "--tag", tag, "--seeds"] + SEEDS + extra, log)
        if not (results / "context_probe_tests_random_window.json").exists():
            run([py, os.path.join(SCRIPTS, "context_probe_tests.py")] + common +
                ["--splits", "graphs_random_window:random_window", "--out", "context_probe_tests_random_window.json"], log)
    if args.baselines_only:
        return

    runs = EXTENDED_GNN_RUNS
    if args.only:  # in the order given
        by_tag = {r[0]: r for r in runs}
        runs = [by_tag[t] for t in args.only if t in by_tag]
    for tag, experiment, graphs, extra, seeds in runs:
        if (results / "gnn_{}.json".format(tag)).exists():
            continue
        if not (work / graphs / "edge_split.npy").exists():
            print("skipping {}: {} not built".format(tag, graphs))
            continue
        run([py, os.path.join(SCRIPTS, "train_gnn.py")] + common +
            ["--experiment", experiment, "--graphs", graphs, "--tag", tag, "--seeds"] + seeds + extra, log)
    print("\nextended steps finished", flush=True)


if __name__ == "__main__":
    main()
