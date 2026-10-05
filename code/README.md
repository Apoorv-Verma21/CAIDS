# CAG-IDS

Context-Aware Graph Intrusion Detection System.

Implementation for the paper *Context-Aware Network Intrusion Detection Using Graph Topology*.

## What it does

Network flows are turned into time-windowed communication graphs, with hosts as
nodes and flows as edges. Each host is given explicit context in three groups:

- **Structural** — in-degree, out-degree, unique peers, PageRank, clustering coefficient, within the current window
- **Behavioural** — z-scores of the host's current activity against its own exponentially weighted history, peer novelty ratio, time since first seen
- **Role** — private or public address, inbound ratio, share of traffic on well-known server ports, destination port entropy

A learned gate decides per host how much to trust topology against context.
All context is computed causally: window *t* uses only windows earlier than *t*.
Evaluation uses a chronological split, not a random one.

## Layout

```
configs/        default.yaml for the real dataset, sample.yaml for fast local tests
notebooks/      run_in_colab.ipynb, the end-to-end runner
scripts/        the command line stages, run in the order listed below
src/cagids/     the library
outputs/        created on first run: work/ for intermediates, results/ for metrics
```

## Install

```
pip install -r requirements.txt
```

On Colab, `torch` is already present and only `xgboost pyyaml pyarrow` need installing.

## Run order

Every script takes `--config` and `--root`. `--root` is the folder that
`data/` and `outputs/` sit inside.

```
python scripts/explore_data.py   --config configs/default.yaml --root .
python scripts/prepare_data.py   --config configs/default.yaml --root .
python scripts/build_graphs.py   --config configs/default.yaml --root .
python scripts/run_baselines.py  --config configs/default.yaml --root .
python scripts/train_gnn.py      --config configs/default.yaml --root . --experiment cagids
python scripts/summarize.py      --config configs/default.yaml --root . --latex
```

Run `explore_data.py` first and read its output. It reports whether every
column was recognised, the class balance, and whether any attack class is
absent from the test split.

## Testing without the real dataset

```
python scripts/make_sample_data.py --out data/sample_flows.csv --rows 40000
python scripts/prepare_data.py  --config configs/sample.yaml --root .
python scripts/build_graphs.py  --config configs/sample.yaml --root .
python scripts/train_gnn.py     --config configs/sample.yaml --root . --experiment cagids --epochs 3
```

The synthetic data exists to prove the pipeline runs. Its numbers mean nothing.

## Experiments

| `--experiment` | Node initialisation | Fusion | Corresponds to |
|---|---|---|---|
| `egraphsage` | vector of ones | none | Lo et al. [1] |
| `centrality_egraphsage` | structural centrality | none | Termos et al. [2] |
| `cagids_concat` | full context | concatenation | ablation for H3 |
| `cagids` | full context | learned gate | proposed |

## Ablations

```
python scripts/build_graphs.py --config configs/default.yaml --root . --no-structural  --tag graphs_no_structural
python scripts/build_graphs.py --config configs/default.yaml --root . --no-behavioural --tag graphs_no_behavioural
python scripts/build_graphs.py --config configs/default.yaml --root . --no-role        --tag graphs_no_role
python scripts/build_graphs.py --config configs/default.yaml --root . --with-time      --tag graphs_with_time
```

Then train against each with `--graphs graphs_<tag> --tag <tag>`.

Window sensitivity uses `--window-seconds 30` or `--window-seconds 300`.

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

## Reproducibility

`--seeds 1 2 3 4 5` runs five seeds and reports mean and standard deviation.
Use that for the numbers that go in the paper.

## References

[1] Lo et al., E-GraphSAGE, NOMS 2022
[2] Termos et al., Centrality-based E-GraphSAGE, IoTBDS 2025
[3] King and Huang, Euler, ACM TOPS 2023
[4] Liu and Guo, DIGNN-A, CMC 2025
[5] Luay et al., Temporal Analysis of NetFlow Datasets, arXiv:2503.04404
