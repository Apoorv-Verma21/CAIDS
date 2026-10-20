# CAG-IDS

Context-Aware Graph Intrusion Detection System.

Implementation for the paper *CAG-IDS: Combining Graph Topology and Historical
Host Context for Flow-Based Network Intrusion Detection, with a Shortcut Audit
of NF-UNSW-NB15-v3*.

## What it does

Network flows are turned into time-windowed communication graphs, with hosts as
nodes and flows as edges. Each host is given explicit context in three groups:

- **Structural** — in-degree, out-degree, unique peers, PageRank, clustering coefficient, within the current window
- **Behavioural** — z-scores of the host's current activity against its own exponentially weighted history, peer novelty ratio, time since first seen
- **Role** — private or public address, inbound ratio, share of traffic on well-known server ports, destination port entropy

A learned gate decides per host how much to trust topology against context.
All context is window-causal: the features of window *t* use window *t* and
earlier windows, never later ones. The behavioural group compares window *t*
with a baseline built from earlier windows only.
Evaluation uses a chronological split, not a random one.

## Layout

```
configs/        default.yaml for the real dataset, sample.yaml for fast local tests
notebooks/      run_in_colab.ipynb, a step-by-step runner for Google Colab
scripts/        the command line stages (see below)
src/cagids/     the library
paper/          the manuscript sources and the Word builder (see "The paper")
outputs/        created on first run: work/ for intermediates, results/ for metrics
```

## Install

```
pip install -r requirements.txt
```

On Colab, `torch` is already present and only `xgboost pyyaml pyarrow` need installing.
The neural models use a CUDA GPU when one is available.

## Running everything

`--root` is the folder that `data/` and `outputs/` sit inside; the dataset CSV
goes to `data/NF-UNSW-NB15-v3.csv`.

```
python scripts/run_all.py --root D:/CAIDS_work --config configs/default.yaml
python scripts/analyze.py --root D:/CAIDS_work
```

`run_all.py` prepares the data, builds every graph variant, runs the dataset
audit, the flat baselines, all neural experiments (five seeds each) and the
context probe tests. It is resumable: finished steps are skipped. `analyze.py`
then writes every results table, the numbers quoted in the paper and the data
figures to `paper/generated/`, including the paired block-bootstrap intervals,
the same-flow comparison of the two split protocols and the gate analysis.

## Robustness checks

```
python scripts/run_all.py --root D:/CAIDS_work --config configs/default.yaml --extended
python scripts/robustness_analysis.py --root D:/CAIDS_work
```

`--extended` builds twelve more graph variants (window-level random split,
session split with flow-level and with window-level validation, the latter also
with an age over active windows or capped at one hour, context ablations under
the random flow split, 30-s and 300-s windows aligned to the 60-s periods) and
runs the experiments listed in `EXTENDED_GNN_RUNS` and `EXTENDED_BASELINE_RUNS`:
seeds 6 to 20 for the main models, for the five neural models under the random
flow split and for gated CAG-IDS without C2 or without the age feature, training
on the forest's 400,000-flow sample, fingerprint-field removal, one-layer,
max-aggregation and direction-aware message passing, window lengths with equal
mini-batches, and the session split without the age feature, with it clipped,
counted over active windows or capped, and with early stopping that ignores
Backdoor and Worms. `--only TAG ...` and `--only-build TAG ...` restrict a call
to some runs or graph builds. The older flag `--revision` still works.
`robustness_analysis.py` writes the numbers the paper quotes from these runs to
`paper/generated/robustness_values.json` and the per-seed strip plot to
`paper/generated/figures/fig_seeds.png`, including the two-level bootstrap (seeds
and test windows) for differences and differences of differences, its versions
with medians and 20% trimmed means over seeds, Hodges–Lehmann shifts,
Bonferroni-adjusted intervals for the planned comparisons, and the prior
correction and per-class offsets tuned on validation together with their spread
over resampled validation sets (cached in `robustness_corrections_cache.json`
next to the results). `paper/literature_search/` documents the literature search
of Section 3.1.

Options of `train_gnn.py` for these checks: `--train-subsample`,
`--windows-per-batch`, `--layers`, `--aggregator {mean,max}`,
`--direction-aware`, `--drop-features`, `--drop-context`, `--clip-context` and
`--early-stop-exclude CLASS ...`. Runs also save validation logits, test logits
and the class weights in their prediction files. `build_graphs.py` has
`--split session` (with `--session-val {flows,windows}`), `--split aligned`,
`--age-mode {elapsed,active}` and `--age-cap N`.

## Single stages

Every script takes `--config` and `--root`.

```
python scripts/explore_data.py         # column resolution and class balance
python scripts/prepare_data.py         # clean the CSV, assign 60-s windows
python scripts/build_graphs.py         # window graphs and causal host context
python scripts/diagnostics.py          # dataset audit: identity, single-feature and context probes
python scripts/run_baselines.py        # random forest and XGBoost
python scripts/train_gnn.py --experiment cagids --seeds 1 2 3 4 5
python scripts/context_probe_tests.py  # why endpoint context hurts a flat classifier
python scripts/summarize.py            # quick text table of all result files
```

## Experiments

| `--experiment` | Node initialisation | Fusion | Role in the paper |
|---|---|---|---|
| `flow_mlp` | none (no graph) | none | B2, flow-only MLP |
| `egraphsage` | vector of ones | none | B3, after Lo et al. [1] |
| `centrality_egraphsage` | structural context | none | B4, after Termos et al. [2] |
| `cagids_concat` | full context | concatenation | gate ablation (H3) |
| `cagids` | full context | learned gate | proposed |
| `cagids_gatv2` | full context | learned gate | attention backbone |
| `context_only` | full context | none, no message passing | context information probe |

The C1 + gate control is `cagids` trained on graphs built with
`--no-behavioural --no-role`.

## Ablations and sensitivity

```
python scripts/build_graphs.py --no-structural  --tag graphs_no_structural
python scripts/build_graphs.py --no-behavioural --tag graphs_no_behavioural
python scripts/build_graphs.py --no-role        --tag graphs_no_role
python scripts/build_graphs.py --with-time      --tag graphs_with_time
python scripts/build_graphs.py --window-seconds 30 --tag graphs_w30
python scripts/build_graphs.py --split random_flow --tag graphs_random_flow
```

Then train against each with `--graphs graphs_<tag> --tag <tag>`. Training
sensitivity uses `--class-weighting sqrt_inverse`, `--loss focal`, or
`--epochs 60 --patience 12`.

## The paper

```
python paper/build_docx.py --out ../paper/CAG-IDS_Final.docx --pdf
```

builds the Word manuscript from `paper/content.md`, `paper/references.json`,
`paper/tables_static.json` and the files that `analyze.py` writes to
`paper/generated/`, using the styles in `paper/template/`. `--pdf` exports a
PDF through Microsoft Word. The builder refuses to run if a quoted value, a
table or a figure is missing, so no number in the paper is typed by hand.

## Things that go wrong

**A required column was not found.** `explore_data.py` prints which role failed to
resolve. Column matching ignores case, spaces and underscores, and knows the
common aliases. Add the real name to `CANONICAL` in `src/cagids/data.py`.

**An attack class is missing from the test split.** Rerun `build_graphs.py` with
`--split blocked`. That assigns contiguous blocks to each split while keeping
time order inside every block.

**A baseline scores above 0.99 macro F1.** That is leakage, not success.
UNSW-NB15 has known leaky features. `data.leaky_features` in the config drops
TTL and per-second byte counts by default. Add more if needed.

**Out of memory.** Lower `train.windows_per_batch`, or set `data.max_windows`
to work on a prefix of the timeline.

## Testing without the real dataset

```
python scripts/make_sample_data.py --out data/sample_flows.csv --rows 40000
python scripts/prepare_data.py  --config configs/sample.yaml --root .
python scripts/build_graphs.py  --config configs/sample.yaml --root .
python scripts/train_gnn.py     --config configs/sample.yaml --root . --experiment cagids --epochs 3
```

The synthetic data exists to prove the pipeline runs. Its numbers mean nothing.

## References

[1] Lo et al., E-GraphSAGE, NOMS 2022
[2] Termos et al., Centrality-based E-GraphSAGE, IoTBDS 2025
[3] Luay et al., Temporal Analysis of NetFlow Datasets, arXiv:2503.04404
