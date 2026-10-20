# Literature search behind Section 3.1

Run on 12 October 2026 against OpenAlex (title and abstract search, works
published from 1 January 2020), with `fetch_openalex.py`:

```
python fetch_openalex.py '<query>' out.json
```

## Queries

| # | Query | Records |
|---|---|---|
| 1 | ("graph neural network" OR "graph neural networks" OR GNN OR GraphSAGE OR "graph convolutional network" OR "graph attention network") AND ("intrusion detection" OR NIDS OR "attack detection") AND (flow OR flows OR NetFlow) | 245 |
| 2 | ("graph neural network" OR "graph neural networks" OR GNN OR GNNs OR GraphSAGE OR "graph convolutional" OR "graph attention") AND ("network intrusion detection" OR "network intrusion" OR "network anomaly detection" OR "malicious traffic") | 424 |
| 3 | ("E-GraphSAGE" OR "edge classification" OR "edge-level") AND ("intrusion" OR "attack" OR "malicious") | 203 |
| 4 | ("graph neural network" OR GNN OR "graph learning") AND ("host behavior" OR "host behaviour" OR "host profiling" OR "behavioral features" OR "behavioural features" OR "node features") AND (traffic OR intrusion OR NetFlow) | 104 |

## Screening

1. 816 distinct records after removing duplicates across the four queries.
2. 559 mention graph learning together with network-traffic terms in the title or abstract.
3. 519 remain after removing, by title, records on vehicular and CAN-bus networks,
   smart grids and power systems, industrial control, provenance graphs, malware,
   log anomalies, blockchain, UAVs and wireless sensor networks
   (`screened_records.csv`).
4. 124 of these mention node features or attributes, host-level, per-host or
   behavioural statistics, historical information or context in the abstract
   (column `abstract_mentions_node_features_or_host_context`). Their abstracts
   were screened; where the node inputs remained unclear, the accessible full
   text was checked.

## What the node inputs turned out to be

Supervised GNN flow classifiers use as node inputs: constant vectors
(E-GraphSAGE and its residual variants), degree or centrality scores (Termos et
al. 2025; Raskovalov et al., arXiv:2212.13994), dataset fields attached to the
endpoints (NE-GConv, Altaf et al., Comput. Secur. 130, 2023: packet-content and
application-level fields on IP:port nodes), statistics of a node's traffic in the
current snapshot (Bala et al., Defence Technology 2026: source dispersion entropy
and Gini concentration), or representations learned from the topology. Temporal
GNNs learn a memory per node. None of the screened papers feeds the classifier a
host's current activity measured against that host's own past.

Limitation: the screen used titles and abstracts, which seldom list node inputs,
so a paper that describes them only in the body can be missed.
