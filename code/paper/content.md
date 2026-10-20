%% Manuscript source. Build with: python code/paper/build_docx.py --out paper/CAG-IDS_Final.docx --pdf
%% {{key}} values come from generated/values.json (scripts/analyze.py); [@key] cites references.json.

@title: CAG-IDS: Combining Graph Topology and Historical Host Context for Flow-Based Network Intrusion Detection, with a Shortcut Audit of NF-UNSW-NB15-v3
@authors: Apoorv Verma, Vidushi Negi, Vihaan Shivesh Vaigankar

@abstract:
Graph neural networks (GNNs) are widely used to classify network flows, but in most of them the hosts carry almost no information, and many evaluations split the flows at random. We present CAG-IDS, which describes every host in every 60-second window with 17 window-causal context features (structural, behavioural against an exponentially weighted history, and role), uses the encoded context as the starting state of an edge-aware GNN and mixes the two through a learned gate. Before comparing models, we audited NF-UNSW-NB15-v3 ({{raw_flows}} flows, ten classes). Four addresses send every attack flow and no benign one, so looking up the source address alone gives a binary F1 of {{probe_src_binary}}. We therefore use ten-class macro-F1 under a chronological split, with five seeds and paired block-bootstrap intervals, over {{n_neural_runs}} neural training runs. A random forest on flow features reaches {{m_rf_macro_mean}}, and the best neural model is a flow-only MLP ({{m_mlp_macro_mean}}). Gated CAG-IDS ({{m_cagids_macro_mean}}) beats an E-GraphSAGE-style GNN by {{boot_cagids_b3_delta}} but not a centrality-initialised GNN. Structural context alone does as well as all 17 features, the gate does not beat concatenation, and behavioural context does not help the rare classes. A random flow split raises macro-F1 for every model, mostly even on identical test flows, and it favours the context-aware models enough to reverse the conclusion, because there the context of a test flow points to the labels of training flows from the same window. On this benchmark, host context mostly identifies hosts and time, and only a chronological evaluation with flat baselines and identity probes makes this visible.

@keywords: Network intrusion detection; Graph neural networks; NetFlow; Host behaviour profiling; Temporal evaluation; Shortcut learning; Dataset audit; UNSW-NB15

# 1. Introduction {#sec:intro}

A network intrusion detection system (NIDS) watches traffic for signs of attack: scanning, denial of service, exploitation, and the quieter steps that come before and after them. Most NIDSs built on machine learning work on flow records such as NetFlow [@claise2004netflow]. A flow record summarises one connection, with its endpoints, ports, protocol, volume, duration and timing [@sarhan2021netflow, @sarhan2022standard, @luay2025temporal]. The usual approach treats each record as a separate row in a table and trains a classifier on those rows. This is cheap and often works well on benchmarks, but it throws away the relationships between flows. If a host suddenly starts contacting forty new destinations, or many hosts start connecting to the same server, no single record shows it.

Graph neural networks (GNNs) can recover some of this information. When hosts become nodes and flows become edges, message passing lets a model combine the features of a flow with those of the flows around it. E-GraphSAGE [@lo2022egraphsage] showed that this works for flow classification. Later papers added residual flow features [@chang2021residuals, @chang2024eressage], self-supervised training [@caville2022anomale], separate node types for hosts and flows [@pujolperich2022unveiling], dynamic snapshots [@duan2023dlgnn, @liu2025dignn] and structural node features [@termos2025centrality]. Two recent surveys cover dozens of such detectors [@bilot2023survey, @zhong2024survey].

Two things about this line of work motivated our study. The first is what a node actually knows. Flow datasets describe edges, not nodes, so in the E-GraphSAGE family every host starts from a constant vector [@lo2022egraphsage] or, at best, from centrality scores [@termos2025centrality]. The model is never told how a host normally behaves. That matters, because the same observation can be harmless for one host and suspicious for another: hundreds of inbound connections a minute are normal for a web server and strange for a printer. Network security has a long tradition of profiling hosts by their own behaviour [@xu2005profiling, @lakhina2005mining, @karagiannis2005blinc, @gu2008botminer], and Kitsune relies on damped per-host statistics [@mirsky2018kitsune]. Yet in the graph-based flow classifiers we reviewed, such profiles were rarely given to the model as node inputs.

The second is how these models are evaluated. Graph-based detectors are often tested on a random split of the flows, so the training set can contain traffic recorded after the test traffic. Temporal splits are recommended for machine learning in security [@arp2022dos, @pendlebury2019tesseract] and for NetFlow data in particular [@luay2025temporal, @elmahdaouy2026contextualized]. Recent reproducibility studies show how much such choices matter. When state-of-the-art graph detectors were re-run under common conditions, their results differed considerably from the published ones, especially in false-positive rates [@wang2025arewe]. In another study, aligning the preprocessing of different systems moved average precision (AP) by up to 61.8 percentage points for a single detector, the snapshot window alone caused a mean relative swing of 38.3% in AP, and a simple control without any graph encoder ranked first on two of four datasets [@zhao2026firstprinciples]. Benchmarks can also contain artefacts that let a model score well for the wrong reasons [@engelen2021troubleshooting, @flood2024smells, @dhooge2022contaminating], which the machine-learning literature calls shortcut learning [@geirhos2020shortcut].

These observations led to the question behind this paper. Does explicit host context, computed only from current and past traffic and mixed with graph topology through a learned gate, help a GNN flow classifier when it is evaluated on later traffic, and does it help the rare attack classes in particular? To answer it, we built CAG-IDS (Context-Aware Graph Intrusion Detection System). CAG-IDS describes every host in every 60-second window with 17 context features and combines their encoding with an edge-aware GNN. Before training any model, we audited the dataset, NF-UNSW-NB15-v3 [@luay2025temporal]. The audit showed that host identity alone separates attacks from benign traffic, which decided our main metric and limits what any host-level method can claim on this benchmark. We then trained {{n_neural_runs}} neural models under one shared protocol and compared them with flat baselines, simple probes and each other. Our contributions are the following.

- **CAG-IDS**, a system that builds a host graph for every window, gives each host 17 window-causal context features in three groups (structural, behavioural with an exponentially weighted history, and role), and fuses the encoded context with an edge-aware GNN through a learned gate.
- **A controlled evaluation framework.** All graph variants share one implementation, and the framework adds flat baselines, information probes, a chronological split, a comparison with a random split on the same flows, ablations of the context groups, sensitivity runs and paired block-bootstrap intervals.
- **An audit of NF-UNSW-NB15-v3.** The benign-versus-attack label is determined by host identity and by TTL, every host in the test period already appears in training, and endpoint context on its own says little about the type of attack.
- **Results under a controlled chronological protocol.** A random forest on flow features (macro-F1 {{m_rf_macro_mean}}) beats every neural model; host context helps a GNN only against constant node features; and a random flow split inflates macro-F1, most of all for the context-aware models, enough to reverse the conclusion about host context.

The rest of the paper is organised as follows. [sec:related] reviews related work, and [sec:gap] states the research gap, our questions and our hypotheses. [sec:method] describes the methodology: the task, CAG-IDS itself and the evaluation measures. [sec:setup] covers the data and the experimental setup, [sec:results] presents the results and [sec:discussion] discusses them. [sec:limitations] lists the limitations and threats to validity, and [sec:conclusion] concludes.

# 2. Background and Related Work {#sec:related}

## 2.1. Flow Records and Flow-Based Detection

Flow records are the most common input for machine-learning NIDSs, because routers can export them cheaply and they do not require looking at packet payloads [@claise2004netflow]. The UNSW-NB15 testbed [@moustafa2015unsw, @moustafa2016evaluation] and the CIC-IDS datasets [@sharafaldin2018cicids] became standard benchmarks, and Sarhan *et al.* re-extracted several of them with one shared NetFlow feature set [@sarhan2021netflow, @sarhan2022standard]. The third release of these datasets adds millisecond timestamps and inter-arrival statistics [@luay2025temporal], which is what makes it possible to process the flows window by window in time order. Random forests [@breiman2001rf] and gradient-boosted trees [@chen2016xgboost] still perform strongly on these data. A model that scores well on one dataset rarely does as well on another, though [@layeghy2023crossdomain], and the gap between benchmark accuracy and real-world usefulness has been known for a long time [@sommer2010closedworld, @apruzzese2023sok].

## 2.2. Graph Neural Networks

Most GNNs can be described as message passing [@gilmer2017mpnn]. In each layer, every edge produces a message from the states of its two endpoints, each node collects the messages that reach it, and the node then updates its state. After *K* layers, the state of a node reflects its neighbourhood up to *K* hops away. Graph convolutional networks [@kipf2017gcn] use a normalised sum of the neighbours' states. GraphSAGE [@hamilton2017graphsage] aggregates the neighbours with a function such as the mean and concatenates the result with the node's own state. It is also inductive: its weights work on graphs it has never seen, which suits network traffic because every window produces a new graph. Graph attention networks [@velickovic2018gat] give each neighbour a learned weight. GATv2 [@brody2022gatv2] changes the order of operations so that the ranking of the neighbours can differ from one receiving node to another, which the original, static form of attention cannot do. To classify flows, edge-aware variants put the edge features into the messages and classify each edge from the representations of its two endpoints.

## 2.3. Graph Neural Networks for Flow Classification

E-GraphSAGE [@lo2022egraphsage] adapted GraphSAGE to aggregate flow features and classify edges, using IP:port endpoints as nodes and a vector of ones as node features. Chang and Branco also fed the flow features straight into the classifier as residuals, which helped the minority classes [@chang2021residuals, @chang2024eressage]. Anomal-E trains a similar encoder without labels [@caville2022anomale]. Pujol-Perich *et al.* treated hosts and flows as two kinds of node and reported that their model stayed robust when packet sizes and inter-arrival times were perturbed [@pujolperich2022unveiling]. Termos *et al.* replaced the constant node vector with degree, betweenness, closeness, PageRank and k-truss scores and reported better detection with fewer false alarms [@termos2025centrality]. Other work handles time explicitly: DLGNN turns snapshots into line graphs for semi-supervised learning [@duan2023dlgnn], and DIGNN-A combines graph convolution, multi-head attention and an integrated dynamic GNN [@liu2025dignn]. Two reproducibility studies are more sceptical. Wang *et al.* found large differences when they re-ran state-of-the-art graph detectors [@wang2025arewe], and Zhao and Hassan found that preprocessing and the snapshot window alone change the results substantially and that a control without a graph encoder is hard to beat [@zhao2026firstprinciples]. [tab:related] compares the systems closest to ours.

@table: related

## 2.4. Temporal Graphs and Host Behaviour Profiling

Another line of work follows how the communication graph changes over time. Euler encodes each snapshot with a GNN, models the sequence of snapshots with a recurrent network and flags unlikely links as lateral movement [@king2023euler]. Bowman *et al.* [@bowman2020lateral], PIKACHU [@paudel2022pikachu], ARGUS [@xu2024argus] and Jbeil [@khoury2024jbeil] use related link-level or temporal-walk approaches, as do general dynamic GNNs such as EvolveGCN [@pareja2020evolvegcn] and temporal graph networks [@rossi2020tgn]. Most of these systems score links without labels. Whatever they know about a host is hidden inside a learned memory rather than written down as features one can inspect.

Profiling hosts by their behaviour is much older than deep learning. Xu *et al.* modelled backbone traffic using the entropy of ports and addresses [@xu2005profiling], and Lakhina *et al.* detected anomalies from changes in the entropy of feature distributions [@lakhina2005mining]. BLINC classified hosts by their social, functional and application-level behaviour [@karagiannis2005blinc], and BotMiner grouped hosts that communicate in similar ways and show similar malicious activity [@gu2008botminer]. Kitsune keeps damped incremental statistics for each host, channel and socket over five decay windows and scores them with autoencoders [@mirsky2018kitsune]. The behavioural part of CAG-IDS follows the same idea. What differs is how the statistics are used: they become node features of a supervised edge classifier, they are computed only from the past, and they are compared with topology-only alternatives inside one implementation. A recent survey calls this general direction contextualised NetFlow-based detection and points to temporal leakage as a common pitfall [@elmahdaouy2026contextualized].

## 2.5. Gated Fusion

The simplest way to combine two representations is to concatenate them and let the next layer work out how to use them. Gated fusion instead learns a mixing weight between the two for every dimension. Highway networks use such gates to mix a transformed input with the original input [@srivastava2015highway], and gated multimodal units use them to combine modalities [@arevalo2017gmu]. In CAG-IDS, a gate mixes a host's neighbourhood embedding with its context embedding. The gate adds no new information. It only changes how the two sources are combined, which is why we compare it directly with concatenation.

## 2.6. Evaluation Pitfalls, Leakage and Shortcuts

Arp *et al.* list common mistakes in machine learning for security, such as data snooping and spurious correlations [@arp2022dos]. TESSERACT shows how ignoring time inflates the scores of malware classifiers [@pendlebury2019tesseract], and Kapoor and Narayanan describe how leakage has undermined results in many fields [@kapoor2023leakage]. For NIDS benchmarks in particular, Engelen *et al.* found labelling and feature-extraction errors in CIC-IDS2017 [@engelen2021troubleshooting], Flood *et al.* describe recurring design smells, that is, flaws in how benchmark datasets are put together [@flood2024smells], and D’hooge *et al.* show that addresses and ports contaminate models [@dhooge2022contaminating]. More generally, Geirhos *et al.* describe how a model can learn a rule that works on the benchmark but not for the task it was meant to solve [@geirhos2020shortcut]. We follow these recommendations and add one more check, an audit of host-identity shortcuts. Such shortcuts matter more for methods that use host context than for methods that look only at flows.

# 3. Research Gap, Research Questions and Hypotheses {#sec:gap}

## 3.1. Research Gap

From this literature we draw three observations. They describe the papers we reviewed, not necessarily the whole field.

First, in the edge-classifying GNNs in [tab:related], the node inputs are constant vectors, structural scores or representations learned from the topology. Per-host behavioural statistics are well established in anomaly detection [@xu2005profiling, @mirsky2018kitsune], but we did not find them used as explicit, past-only node inputs to supervised GNN flow classifiers. Whether they help is therefore an open empirical question.

Second, the effect of the evaluation protocol on graph detectors is well documented [@wang2025arewe, @zhao2026firstprinciples]. Host context brings an extra risk. Any feature that summarises a host over a period that includes the test window, or later windows, leaks information from the future into the prediction.

Third, dataset audits usually look at flow features [@engelen2021troubleshooting, @flood2024smells, @dhooge2022contaminating]. For a method that describes hosts, the more important question is how much of the label host identity alone explains.

CAG-IDS is built from existing pieces: message passing, host profiling, exponentially weighted statistics, gated fusion and chronological evaluation. None of them is new. Our contribution is to put them together in one edge-classifying system with a strictly causal implementation, to test in a controlled way whether the combination helps, and to audit what the benchmark can and cannot tell us.

## 3.2. Research Questions and Hypotheses {#sec:hypotheses}

**RQ1 (dataset audit).** How much of the label of NF-UNSW-NB15-v3 can be explained by host identity and by fields that reflect the testbed rather than attacker behaviour, and what does this mean for evaluating host-context methods?

**RQ2 (evaluation protocol).** How much does a random flow split change the measured performance compared with a chronological split, for flat models and for graph models?

**RQ3 (host context).** Does window-causal host context, fused with topology through a learned gate, improve a GNN flow classifier under chronological evaluation, especially for rare attack classes?

We judge RQ3 by macro-F1 and attack macro-F1 only, because the false-positive rate leaves no room to separate models on this dataset ([sec:audit]). To make RQ3 testable, we wrote down three hypotheses and their acceptance rules before training any GNN model.

**H1.** CAG-IDS reaches a higher test macro-F1 than both the E-GraphSAGE-style baseline and the centrality-initialised baseline on the same chronological split. We accept H1 for a baseline if the 95% block-bootstrap interval of the macro-F1 difference ([sec:metrics]) lies entirely above zero.

**H2.** Behavioural context helps most for the low-volume attack classes, by which we mean classes with less than 4% of the attack flows: Analysis ({{share_analysis}}), Backdoor ({{share_backdoor}}), Shellcode ({{share_shellcode}}) and Worms ({{share_worms}}). H2 is supported if removing the behavioural group lowers the mean F1 of these four classes more than removing the structural group or the role group does, and by more than the seed-to-seed standard deviation. We report Reconnaissance ({{share_reconnaissance}} of attack flows) separately, because scanning is the behaviour that host profiles are most often expected to catch.

**H3.** Gated fusion beats plain concatenation of the topological and context embeddings, under the same acceptance rule as H1.

# 4. Methodology {#sec:method}

This section defines the task, describes CAG-IDS and the variants we compare, and fixes the measures used to evaluate them. All of the paper's formulas appear in this section.

## 4.1. Problem Formulation {#sec:problem}

A NetFlow record *f* = (*u*_{*f*}, *v*_{*f*}, *τ*_{*f*}, **e**_{*f*}, *y*_{*f*}) has a source host *u*_{*f*} and a destination host *v*_{*f*} (both IPv4 addresses), a start time *τ*_{*f*}, a feature vector **e**_{*f*} and a label *y*_{*f*}. The label is one of the ten classes in 𝒞, that is, Benign or one of nine attack categories. With window length Δ and *τ*_{0} the first start time in the capture, flow *f* falls into window

@eq: window {#eq:window}

The graph of window *t* is the directed multigraph *G*_{*t*} = (*V*_{*t*}, *E*_{*t*}). Its nodes are the hosts that send or receive at least one flow in the window, and its edges are the window's flows, each carrying its feature vector. When two hosts exchange several flows, each flow stays a separate edge and gets its own prediction.

The task is to assign every flow of window *t* to one class in 𝒞. We call a detector *window-causal* if its predictions for window *t* depend only on flows that started before the end of window *t*. This fits a micro-batch deployment: the detector collects one window of flows, classifies them all when the window closes, and moves on. The delay before a decision is therefore at most Δ plus the processing time. Window causality concerns windows, not single flows, since a flow is only classified after the later flows of its window have arrived. All context in CAG-IDS is window-causal, and the behavioural features go one step further and depend only on windows strictly before *t*.

We assume a passive monitor that sees NetFlow records at a single vantage point and has a labelled history to train on. Attackers control the traffic they send but cannot touch the monitor or its labels. The task is supervised classification into known classes. We do not claim to detect new kinds of attack, and an adaptive attacker who tries to evade the system is outside our scope ([sec:deployment]).

## 4.2. Overview of CAG-IDS {#sec:overview}

[fig:architecture] gives an overview of the pipeline, and Algorithm 1 shows the exact order in which our code builds the graphs and the context. For each window, CAG-IDS builds the graph *G*_{*t*}. It then computes a context vector **c**_{*v*} for every active host, using *G*_{*t*} and a per-host history ℋ_{*t*−1}, and encodes it as **c**′_{*v*}. This encoding is the starting point for *K* layers of edge-aware message passing. A gate then mixes the output of the last layer with **c**′_{*v*}, and a classifier labels each flow from the fused representations of its two endpoints together with the flow's own features.

In our experiments, the graphs and the context are computed once, offline, before any model is trained. This gives the same result as the deployment order, in which the history would be updated after each window has been classified, because the context of window *t* never depends on the classifier and never includes window *t* or anything after it.

@fig: figures/fig1_architecture.png | 6.27 | Overview of CAG-IDS. Each 60-s window becomes a directed multigraph. Host context comes from the window itself (C1, C3) and from a history of earlier windows only (C2), which is updated after the window's context has been computed. The encoded context initialises the message passing and is fused with its output through a learned gate; the standardised flow features enter both the messages and the classifier. {#fig:architecture}

@algorithm: algorithm1

## 4.3. Window Graph Construction {#sec:windows}

By default a window is Δ = 60 s long; the sensitivity runs also use 30 s and 300 s. Windows with fewer than two flows are skipped. Node indices are local to each window, so a host that is active in many windows appears as a separate node in each of them, and anything that carries over from one window to the next has to go through the history. Parallel flows between the same two hosts stay separate edges, so a busy window produces a small but dense multigraph.

## 4.4. Host Context {#sec:context}

[tab:context] lists the 17 context features. Each group answers a different question about a host. C1 asks where the host sits in the current graph, C2 how its current activity compares with its own past, and C3 what role it seems to play. An optional fourth group, C4, encodes the time of day; we use it only to check for shortcuts based on the schedule of the capture.

@table: context

### 4.4.1. Structural Context (C1)

The structural group has five features: the in-degree, the out-degree and the number of distinct peers (each as log(1 + *x*)), the PageRank [@page1999pagerank] and the local clustering coefficient [@watts1998smallworld]. We compute PageRank by power iteration on the directed multigraph, so two hosts linked by many flows count for more than two hosts linked by one:

@eq: pagerank {#eq:pagerank}

Here *n* = |*V*_{*t*}|, *d* = 0.85, *w*_{*uv*} is the number of flows from *u* to *v*, *w*_{*u*} is the number of flows sent by *u*, and 𝒟 is the set of hosts that send no flows, whose rank is spread evenly over all hosts. The iteration starts from a uniform vector, runs for 30 steps and renormalises the vector after each step. We use *n* *r*_{*v*} as the feature, which has a mean of one in every window. The clustering coefficient is computed on the undirected simple graph behind *G*_{*t*}, with parallel flows merged and self-loops removed:

@eq: clustering {#eq:clustering}

where *T*_{*v*} is the number of triangles that contain *v* and *k*_{*v*} is its number of distinct neighbours. *C*_{*v*} is zero when *k*_{*v*} < 2. These features are a subset of the centrality measures used by Termos *et al.* [@termos2025centrality]; we do not compute betweenness, closeness or k-truss.

### 4.4.2. Behavioural Context (C2)

The behavioural group compares what a host is doing now with what it usually does. For each host *v* that is active in window *t*, we compute four statistics:

@eq: stats {#eq:stats}

where *n*_{*v*} is the number of flows the host sends or receives, *B*_{*v*} their total bytes (IN_BYTES plus OUT_BYTES), *Q*_{*v*} the set of destination ports it contacts and *P*_{*v*} its set of peers. For each statistic *j*, the history stores an exponentially weighted moving average (EWMA) *μ*_{*v,j*} and an exponentially weighted variance *s*_{*v,j*}. The context for window *t* contains the clipped z-scores

@eq: zscore {#eq:zscore}

computed from the history as it was before window *t*, with *ε* = 10^{−6} and *τ* = 10. A large positive value means that the host is much busier than usual, or is contacting many more ports or peers than usual. Once the context of window *t* has been computed, we update the history of every active host with *δ*_{*v,j*} = *x*_{*v,j*} − *μ*_{*v,j*}, using the incremental form given by Finch [@finch2009incremental]:

@eq: ewma_mean {#eq:ewma_mean}

@eq: ewma_var {#eq:ewma_var}

The first time a host is seen, its mean is set to that first observation and its variance to one (in log units). A host's history is only updated in windows where the host is active. With *α* = 0.3, the weight of an observation halves roughly every two such windows, and the last seven windows carry about 92% of the total weight (1 − 0.7^{7}). In practice, the baseline therefore follows the last few minutes of a host's activity.

The peer-novelty ratio

@eq: novelty {#eq:novelty}

is the fraction of the host's current peers *P*_{*v*}^{*t*} that are not in the set *S*_{*v*}^{*t*−1} of peers seen in earlier windows. The age feature is log(1 + *a*), where *a* is the number of 60-s intervals since the host was first seen. Finally, a first-seen flag marks hosts that have no history yet. For these hosts the z-scores and the age are set to zero and the novelty to one.

### 4.4.3. Role Context (C3)

The role group describes what kind of host *v* appears to be. It contains a flag for private or special-purpose addresses (as defined by Python's ipaddress module, which covers RFC 1918 [@rekhter1996rfc1918] and other reserved blocks), the inbound ratio *d*_{in}/(*d*_{in} + *d*_{out}), the share of the flows started by the host that go to a well-known service port, log(1 + *d*_{in} + *d*_{out}), and the Shannon entropy of the destination ports the host contacts,

@eq: entropy {#eq:entropy}

where *p*_{*v,k*} is the fraction of its outgoing flows that go to port *k*. The 34 service ports are 20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123, 137, 138, 139, 143, 161, 389, 443, 445, 465, 514, 587, 631, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 8080 and 8443. The idea is simple. A scanner contacts many different ports and starts most of its own flows, while a server mostly receives flows on a handful of ports. The optional group C4 adds the sine and cosine of the window's start time. It is not part of CAG-IDS; if adding it improved the results, that would point to a shortcut based on the schedule of the capture.

### 4.4.4. State, Causality and Normalisation {#sec:causality}

The context for window *t* is always computed before the history is updated with window *t*. The structural and role features do use all flows of window *t*, which window causality allows ([sec:problem]). A single history runs through the training, validation and test periods in time order, just as it would in a deployment. This has two side effects. A host's baseline includes any attack traffic it produced earlier ([sec:deployment]). And the history is carried over gaps in the capture without any decay, while the age feature keeps counting the elapsed intervals, gap included. Before encoding, the raw context values are standardised with the mean and standard deviation over all host-window pairs in the training windows. The probes in [sec:audit] use the raw values.

## 4.5. Model {#sec:model}

### 4.5.1. Context Encoder and Edge-Aware Message Passing

A two-layer perceptron (128 hidden units, ReLU, dropout) maps the standardised context **c**_{*v*} ∈ ℝ^{17} to **c**′_{*v*} ∈ ℝ^{64}. This vector is the initial node state **h**_{*v*}^{(0)}, and it is also one of the two inputs to the gate. In layer *k* of message passing, each flow produces a message from the sender's state and the flow's features, each host averages the messages it receives, and then it updates its state:

@eq: message {#eq:message}

@eq: aggregate {#eq:aggregate}

@eq: update {#eq:update}

Here 𝒩(*v*) is the multiset of neighbours of *v* over both incoming and outgoing flows, so three parallel flows to the same peer give three messages. Each flow sends a message to both of its endpoints with the same weights, so a host learns both from what it sends and from what it receives. The direction of the flow is not lost, because the classifier keeps the source and destination representations in order. Dropout is applied after every ReLU. With *K* = 2 layers of 128 units, each host's state summarises its two-hop neighbourhood, which in a window of a few dozen hosts already covers much of the graph.

### 4.5.2. Attention Backbone

As an alternative backbone, we replace the mean in Eq. [eq:aggregate] with GATv2-style attention over flows [@brody2022gatv2], using four heads of 32 units. For a flow from *u* to *v*, the score, the attention weight and the aggregate are

@eq: att_score {#eq:att_score}

@eq: att_weight {#eq:att_weight}

@eq: att_aggregate {#eq:att_aggregate}

The softmax runs over all flows that reach *v*, each head *q* has its own projections, and the outputs of the four heads are concatenated. Because the value of each message includes the projected flow features, the attention can weight flows by what they contain and not only by who sent them. Dropout is applied to the attention weights, and the update step is the same as in Eq. [eq:update].

### 4.5.3. Gated Fusion

After the last layer, we combine the topological embedding **h**_{*v*}^{(*K*)} and the context embedding **c**′_{*v*} through a gate that is computed separately for each host and each dimension:

@eq: gate {#eq:gate}

@eq: fusion {#eq:fusion}

Here *σ* is the logistic function, ⊙ is element-wise multiplication, and *W*_{*c*} and **b**_{*c*} project the 64-dimensional context embedding to 128 units. A gate value close to one means that, in that dimension, the representation of *v* comes mostly from its neighbourhood; a value close to zero means it comes mostly from the host's own context. Because the gate looks at both inputs, it can rely on the neighbourhood for some hosts and on the context for others. Our expectation was that hosts with few flows, whose neighbourhood offers little evidence, would lean on their context. The concatenation variant uses **z**_{*v*} = [**h**_{*v*}^{(*K*)} ‖ **c**′_{*v*}] instead, which gives the classifier the same information but no learned mixing weight.

### 4.5.4. Flow Classifier and Training Objective

Each flow is classified from the fused representations of its two endpoints and its own features,

@eq: classifier {#eq:classifier}

where the MLP has one hidden layer of 128 units. Feeding **e**_{*uv*} directly into the classifier means that parallel flows between the same two hosts can still get different labels [@chang2024eressage, @termos2025centrality]. We train by minimising the class-weighted cross-entropy over the flows *f* in each mini-batch ℬ,

@eq: loss {#eq:loss}

using inverse-frequency class weights computed once from the training flows,

@eq: class_weight {#eq:class_weight}

where *N* is the number of training flows, *N*_{*c*} the number in class *c*, and *C* = 10. Without this weighting, Worms ({{share_worms}} of attack flows) would contribute almost nothing to the loss. As a planned sensitivity run, we also train gated CAG-IDS with focal loss [@lin2017focal] with *γ* = 2. After the planned runs, we added square-root inverse-frequency weights, focal loss for the other variants and a longer training budget as post-hoc checks ([sec:sensitivity]).

## 4.6. Controlled Variants {#sec:variants}

[tab:variants] lists the variants we compare. All graph variants share the windowing, the context pipeline, the message-passing code, the classifier head, the loss and the training loop. They differ only in how the nodes are initialised, how the two embeddings are fused, which backbone is used and which context groups the graphs carry.

@table: variants

The flow-only MLP (B2) sees neither the graph nor the context, so comparing it with the graph models separates the effect of the graph from the effect of using a neural classifier. The E-GraphSAGE-style baseline (B3) gives every node the same constant vector, as E-GraphSAGE does [@lo2022egraphsage]. It differs from the original in using hosts as nodes, one graph per window and the residual flow input, so it is a baseline in the style of E-GraphSAGE and not a reproduction of it. The centrality-initialised baseline (B4) encodes only the C1 features. It follows the idea of Termos *et al.* [@termos2025centrality] but not their full feature set. The concatenation variant isolates the effect of the gate (H3), and the GATv2 variant the effect of the aggregation function. The context-only probe measures how much label information the encoded context of the two endpoints carries on its own, without message passing or flow features.

B3 and B4 have no fusion step, so their classifier only sees **h**^{(*K*)}, while CAG-IDS also sees the context embedding directly. Comparing CAG-IDS with B4 therefore changes two things at once: the context groups and the direct path from the context to the classifier. The C1 + gate control removes this confound. It is gated CAG-IDS trained on graphs whose context holds only C1, so it shares B4's information and CAG-IDS's architecture. The models also differ in size, from {{params_flow_mlp}} parameters for B2 to {{params_cagids}} for gated CAG-IDS and {{params_cagids_gatv2}} for the GATv2 variant.

Building the inputs is cheap. For a window with *n* hosts and *m* flows, the degrees and peer sets cost *O*(*m*), PageRank *O*(*Im*) for *I* iterations, triangle counting *O*(Σ_{*v*} *k*_{*v*}^{2}), and the behavioural and role features *O*(*m* + *n*). The history needs memory proportional to the number of hosts seen so far. One GNN layer costs *O*(*m* *h*^{2}) for hidden size *h*, since every flow produces two messages.

## 4.7. Evaluation Metrics and Statistical Analysis {#sec:metrics}

For each class *c*, precision, recall and F1 are

@eq: prf {#eq:prf}

and our main metric is the ten-class macro-F1,

@eq: macro_f1 {#eq:macro_f1}

where 𝒞 contains the classes present in the split being evaluated (all ten in the chronological test period). Macro-F1 gives the {{test_count_worms}} Worms test flows the same weight as the {{test_count_benign}} benign ones. Because host identity alone can decide benign versus attack ([sec:audit]), we also report the macro-F1 over the nine attack classes, which we call attack macro-F1. As secondary metrics we report weighted F1, binary F1 after merging all attack classes, and the false-positive rate on benign flows,

@eq: fpr {#eq:fpr}

where FP counts benign test flows labelled as some attack class and TN counts those labelled correctly. We also report the average precision (AP) of the attack score 1 − *p*(Benign) [@saito2015precision], the per-class F1, per-class precision and recall, a confusion matrix and the processing time.

Running several seeds captures the randomness of training, but not the randomness of which test period we happen to have. For paired comparisons we therefore use a moving-block bootstrap over test windows [@kunsch1989jackknife]. Blocks of ten consecutive windows (10 min) are drawn with replacement until we have a resampled test period *I** of the original length; using blocks keeps the short-range dependence between neighbouring windows. On each of 2,000 replicates, we recompute the macro-F1 of every seed of both models, average it over the *S* = 5 seeds, and record the difference

@eq: bootstrap {#eq:bootstrap}

The 2.5th and 97.5th percentiles of Δ* give the 95% interval. The same procedure, restricted to a common set of flows, compares models trained under different split protocols. We report means and sample standard deviations over seeds and do not use significance stars. With only five seeds, a two-sided Wilcoxon test cannot go below *p* = 0.0625, so it would be of little use here. We do not correct the three planned comparisons (H1 against two baselines, and H3) for multiple testing.

# 5. Dataset and Experimental Setup {#sec:setup}

## 5.1. NF-UNSW-NB15-v3 {#sec:dataset}

We use NF-UNSW-NB15-v3 [@luay2025temporal], the third NetFlow release of the UNSW-NB15 testbed [@moustafa2015unsw], downloaded from the University of Queensland data repository. The SHA-1 digest of the CSV file matches the repository manifest. The file contains {{raw_flows}} flows with 53 NetFlow fields, a binary label and an attack category, split into {{raw_benign}} benign and {{raw_attack}} attack flows ({{attack_share}}). The dataset description gives 127,639 attack flows next to the same benign and total counts, but these numbers do not add up, so we report the counts we found in the file. The capture has two sessions, 22 January 2015 ({{s1_flows}} flows over {{s1_hours}} h) and 18 February 2015 ({{s2_flows}} flows over {{s2_hours}} h), {{session_gap_days}} days apart, and their attack rates differ widely ({{s1_attack_rate}} and {{s2_attack_rate}}). There are {{n_hosts}} addresses in total, {{n_private_hosts}} of them private or special-purpose. Most benign traffic runs between a group of clients and a group of servers, while the attack traffic goes from a few attacker addresses to a separate group of victim servers. This structure turns out to matter a great deal ([sec:audit]).

Windows with fewer than two flows are dropped, which removes {{flows_dropped}} flows and leaves {{n_flows}} flows in {{n_windows}} non-empty 60-s windows. A window contains a median of {{flows_per_window_median}} flows (at most {{flows_per_window_max}}) among a median of {{hosts_per_window_median}} hosts (at most {{hosts_per_window_max}}). The graphs are therefore small and dense, with many parallel flows between the same hosts. Across the {{session_gap_days}}-day gap between the sessions the history is carried over unchanged, and the log-age feature reaches at most {{max_log_age}}.

## 5.2. Flow Features and Preprocessing {#sec:features}

[tab:features] summarises how we use the NetFlow fields. Addresses, ports and timestamps are used to build the graph and the context, but we remove them from the flow vector so that no classifier can simply memorise endpoints [@dhooge2022contaminating]. We also remove MIN_TTL and MAX_TTL, because their values most likely reflect the testbed (its operating systems and routing) rather than what the attacker does. The two per-second byte rates are removed as well, since they are infinite for zero-duration flows ({{inf_rate_flows}} records). [sec:audit] shows how much label information these removed fields carry. Heavy-tailed fields are transformed with log(1 + *x*) after negative values are clipped to zero, and every feature is standardised with statistics from the training period. Categorical codes (L7_PROTO, ICMP_TYPE, DNS_QUERY_TYPE) and the identifier DNS_QUERY_ID are kept as plain numbers, as in the released feature set; we list this as a limitation.

@table: features

## 5.3. Splitting Protocols {#sec:splits}

***Chronological split (primary).*** We sort the {{n_windows}} windows by time and split them 70/10/20 into {{n_train_windows}} training, {{n_val_windows}} validation and {{n_test_windows}} test windows. Training covers the whole first session and the first {{train_s2_hours}} h of the second (until {{train_end}} UTC on 18 February). Validation covers the next {{val_hours}} h, and testing the last {{test_hours}} h ({{test_start}} to {{test_end}} UTC). Every class appears in every split ([tab:dataset], [fig:timeline]), so the blocked split that the code offers as a fallback was not needed. Only {{test_clock_overlap}} of the test flows fall in a minute of the day that also occurs in training.

@fig: figures/fig2_timeline.png | 6.27 | Flows per 10-min bin over the two capture sessions (symmetric log scale). Grey line: benign flows; stacked bars: attack classes. Shaded areas mark the validation and test periods of the chronological split; everything before them is training data. The attack mix and intensity differ between the sessions and within the second session. {#fig:timeline}

@table: dataset

***Random flow split (comparison only).*** Here the flows are assigned to training, validation and test at random within each class (70/10/20), while every window graph keeps all of its flows for message passing. The loss and the metrics use only the flows of the relevant subset, and context is still computed causally. This copies the random-split protocol that is common in the GNN literature, and we use it only for RQ2. Because its test set is spread across the whole capture, it contains different traffic from the chronological test set; for example, it has {{rnd_test_analysis}} Analysis flows, against {{test_count_analysis}} in the chronological test set. To separate the protocol from the test set, we also score both kinds of model on the flows that the two test sets share: the random-split test flows that fall in the chronological test period ({{same_flow_n}} flows).

## 5.4. Baselines and Probes {#sec:baselines}

**Flat baselines (B1).** We use a random forest [@breiman2001rf] with 200 trees and class weights balanced within each bootstrap sample, and XGBoost [@chen2016xgboost] with 200 trees of depth 6 and inverse-frequency sample weights. Both use the 46 flow features. To keep training time reasonable, their training set contains every attack flow from the training period, plus enough randomly chosen benign training flows to reach 400,000 flows. The neural models train on all {{n_train_flows}} training flows. The neural baselines B2 to B4 are described in [sec:variants].

We also use three kinds of simple probe. **Identity probes** predict, for each test flow, the most common training class of a key (the source address, the destination address, the destination port, the address pair, or the address pair together with the destination port), and Benign for keys that never appeared in training. **Single-feature probes** are depth-6 decision trees with balanced class weights, each trained on a single feature using all training attack flows and 300,000 benign training flows, and tested on the full test split. **Context probes** are 200-tree random forests trained on the raw context of the two endpoints (34 values), on the flow features, and on both together, using the same 400,000-flow sample as B1 and three seeds. Further forest probes replace the endpoint context with a row-shuffled copy, drop or keep only the behavioural group, or use the random split; they test why context changes the forest's results ([sec:contexttests]).

## 5.5. Implementation and Training {#sec:training}

We implemented the neural models in PyTorch [@paszke2019pytorch] without a graph library, using index-based scatter operations, and the flat baselines with scikit-learn [@pedregosa2011sklearn] and XGBoost. Training uses Adam [@kingma2015adam] on mini-batches of eight windows, which are merged into one disconnected graph and shuffled every epoch. We stop early based on validation macro-F1, with a patience of six epochs, and restore the best epoch. All settings ([tab:settings]) were fixed before the runs and are the same for every model; we did not tune any model individually. This is fair to all variants, but it also means that none of them is tuned to its best, which is why we report the sensitivity runs in [sec:sensitivity]. Every configuration is trained with seeds 1 to 5, which set the NumPy and PyTorch random generators and, for the flat baselines, also choose the benign sample. The flat baselines and probes ran on a laptop CPU (AMD Ryzen 7 7840HS, 16 GB) with Python 3.13, scikit-learn 1.9 and XGBoost 3.4, and the neural models on the laptop's NVIDIA GeForce RTX 4060 GPU with PyTorch 2.11.

@table: settings

## 5.6. Experiments {#sec:inventory}

[tab:inventory] lists every experiment, what it is for and where its results appear. All of them were run to completion.

@table: inventory

# 6. Results {#sec:results}

## 6.1. Dataset Audit (RQ1) {#sec:audit}

We ran the audit before comparing any models. [tab:diagnostics], [fig:single] and [fig:context] summarise it.

***Host identity determines the binary label.*** All {{raw_attack}} attack flows come from {{n_attack_sources}} addresses (175.45.176.0 to 175.45.176.3), and none of these addresses ever sends a benign flow. Their targets, the {{n_victims}} servers 149.171.126.10 to 149.171.126.19, receive only {{benign_to_victims}} benign flows in the entire capture. Meanwhile, {{benign_core_share}} of all benign flows run between the clients 59.166.0.0 to 59.166.0.9 and the servers 149.171.126.0 to 149.171.126.9. Simply looking up the most common training class of each source address gives a binary F1 of {{probe_src_binary}} on the chronological test split, with a false-positive rate of {{probe_src_fpr}}; the destination address alone gives {{probe_dst_binary}}. These lookups say almost nothing about the type of attack (ten-class macro-F1 {{probe_src_macro}}), because each attacker launches every category in similar proportions: each one sends between {{attacker_analysis_min}} and {{attacker_analysis_max}} Analysis flows and between {{attacker_worms_min}} and {{attacker_worms_max}} Worms flows. The test period also contains no new hosts. Of the test attack flows, {{test_attack_from_train_sources}} come from addresses that already attacked during training, and all {{hosts_test}} hosts active in the test period also appear in training. One more detail is easy to miss. Flows sent back to an attacker are labelled benign, so the attackers are the destination of {{benign_to_attackers}} benign flows ({{benign_victim_to_attacker}} of them from the victims), even though they never send one. The label therefore partly encodes the direction of a connection.

@table: diagnostics

***TTL is a testbed artefact.*** A depth-6 tree on MIN_TTL alone reaches a binary F1 of {{probe_ttl_binary}}, and one on MAX_TTL reaches {{probe_maxttl_binary}} ([fig:single]). Benign flows have a minimum TTL of 31 in {{ttl_benign_31}} of cases, while attack flows have 254 ({{ttl_attack_254}}) or 62 ({{ttl_attack_62}}). These values most likely come from the hosts' operating systems and their position in the testbed. The excluded byte rates carry much less information (binary F1 {{probe_rate_s2d_binary}} and {{probe_rate_d2s_binary}}), so removing them was a precaution more than a necessity. Among the features we kept, the best single-feature tree reaches a ten-class macro-F1 of only {{probe_best_kept_macro}} ({{probe_best_kept_name}}). Some packet-size fields still separate benign from attack traffic well on their own, with a binary F1 of {{probe_shortest_binary}} for SHORTEST_FLOW_PKT and {{probe_minlen_binary}} for MIN_IP_PKT_LEN. Even without addresses, ports and TTL, then, the binary task stays easy, while no single feature identifies the type of attack.

***Context alone carries little attack-type information.*** A random forest trained only on the raw context of the two endpoints, with no flow features at all, reaches a ten-class macro-F1 of {{context_only_rf_macro}} but a binary F1 of {{context_only_rf_binary}}. [fig:context] shows why the binary score is so high. The other hosts fall into two groups, hosts that send almost nothing and busy clients with a high out-degree and high port entropy, and the attackers sit in between. {{band_attackers}} of the attacker host-windows lie between the attackers' own 5th and 95th percentiles on both out-degree and port entropy, compared with only {{band_others}} ({{band_others_n}}) of the other host-windows. In a closed testbed with only {{n_hosts}} addresses, the context of the two endpoints thus works as a fingerprint of the attacking machines. Some of the features we designed turned out to be uninformative here. Peer novelty is zero in {{novelty_zero_share}} of the {{host_window_pairs}} host-window pairs (only {{first_seen_pairs}} are first sightings), and the behavioural z-scores of the attackers and of the other hosts overlap heavily.

@fig: figures/fig3_single_feature.png | 4.3 | Single-feature probes: a depth-6 tree is trained on one feature and scored on the chronological test split. Asterisks mark the four fields excluded from all models. TTL alone separates benign from attack flows; no feature identifies the attack type on its own. {#fig:single}

@fig: figures/fig4_context_separation.png | 4.2 | Distribution of four raw context features over all host-window pairs: the four attack sources (red) and all other hosts (teal). Out-degree and destination-port entropy place the attackers between the two groups of other hosts; peer novelty is almost always zero in this closed testbed, and the z-scores overlap. {#fig:context}

Adding endpoint context to the flow features actually *lowers* the random forest's macro-F1, from {{flow_only_rf_macro}} to {{flow_plus_context_rf_macro}}, and its attack macro-F1 from {{flow_only_rf_attack}} to {{flow_plus_context_rf_attack}}. [sec:contexttests] examines why. For the rest of the paper we treat binary F1, the false-positive rate and AP only as sanity checks, and we base our conclusions on ten-class and attack macro-F1.

## 6.2. Main Comparison (H1 and H3) {#sec:main}

[tab:main] lists the detection performance of all models on the chronological test split, and [tab:significance] gives the paired comparisons. Every model separates benign from attack traffic almost perfectly: binary F1 is at least {{main_min_binary}}, AP at least {{main_min_ap}} and the false-positive rate at most {{main_max_fpr}}. The differences that matter are in the attack types.

@table: main

The random forest is the strongest model by a wide margin, with a macro-F1 of {{m_rf_macro}}, followed by XGBoost ({{m_xgb_macro}}). The best neural model is the flow-only MLP ({{m_mlp_macro}}), which sees no graph at all, and every graph model scores below it. Gated CAG-IDS trails the random forest by {{boot_cagids_rf_delta_abs}} ({{boot_cagids_rf}}). The neural context-only probe reaches {{m_ctxonly_macro}}, only slightly above the forest trained on the same raw context ({{context_only_rf_macro}}), so the context of the two endpoints alone says little about the type of attack, whichever model reads it.

@table: significance

***H1 holds against B3 but not against B4.*** Gated CAG-IDS ({{m_cagids_macro}}) beats the E-GraphSAGE-style baseline ({{m_b3_macro}}) by {{boot_cagids_b3}}, an interval entirely above zero. Against the centrality-initialised baseline the result is reversed: B4 reaches {{m_b4_macro}}, and the difference of {{boot_cagids_b4}} lies entirely below zero. Attack macro-F1 shows the same pattern. The C1 + gate control explains the reversal. It reaches {{m_c1gate_macro}}, the same as gated CAG-IDS ({{boot_cagids_c1gate}}), and falls below B4 by the same margin ({{boot_c1gate_b4}}). Adding the behavioural and role groups to C1 therefore changes nothing, and the deficit against B4 comes from the architecture, that is, from the gate and the direct path from the context to the classifier, not from the context itself. The gap to B4 is small, though, and it is not stable across window lengths and training settings ([sec:sensitivity]).

***H3 is not supported.*** Gating beats concatenation by {{boot_cagids_concat}}. The lower end of the interval lies marginally below zero, so by our acceptance rule the gate is not better than concatenation, although the margin is small. The gate did make the runs more consistent: the seed standard deviation of macro-F1 is {{m_cagids_macro_sd}} with the gate and {{m_concat_macro_sd}} with concatenation. Replacing the mean aggregation with GATv2 attention changes nothing either ({{boot_gatv2_cagids}}).

***Message passing does not help here.*** The E-GraphSAGE-style baseline falls {{boot_b3_mlp_delta_abs}} below the flow-only MLP ({{boot_b3_mlp}}), and even the best graph model, B4, stays below it ({{boot_b4_mlp}}). Starting the nodes from context instead of a constant vector recovers part of this loss but not all of it ({{boot_cagids_mlp}} for gated CAG-IDS against the MLP). The best epoch lies well before the limit of 30 in most runs (median {{best_epoch_median_all}} over {{best_epoch_runs}} runs), although {{best_epoch_ge25}} runs peaked in the last six epochs; [sec:sensitivity] checks whether longer training changes the ranking.

## 6.3. Per-Class Results {#sec:perclass}

[tab:perclass] breaks the scores down by class. Every model recognises Generic, Exploits, Fuzzers and Reconnaissance reasonably well and does poorly on Backdoor and DoS. The neural models do better than the forest on Analysis (F1 {{pc_cagids_analysis}} for gated CAG-IDS against {{pc_rf_analysis}}), but they lose heavily on Worms, Shellcode, DoS and Exploits. On Worms, for example, the forest reaches {{pc_rf_worms}} and gated CAG-IDS {{pc_cagids_worms}}.

@table: perclass

[tab:rareclass] shows where these losses come from. On the weakest classes, the neural models combine high recall with very low precision. On Worms, gated CAG-IDS reaches a recall of {{rec_cagids_worms}} but a precision of only {{prec_cagids_worms}}, and the flow-only MLP behaves the same way (recall {{rec_mlp_worms}}, precision {{prec_mlp_worms}}). The same holds for Analysis and Shellcode. With the class weights of Eq. [eq:class_weight], a Worms flow counts about {{weight_ratio_worms_benign}} times as much as a benign flow in the loss, so the networks learn to predict the rare classes generously, and a small share of false alarms from the large classes is enough to swamp the rare classes' few true positives. The forest, trained with balanced weights on a subsample, is far more conservative (Worms precision {{prec_rf_worms}}, recall {{rec_rf_worms}}). [sec:sensitivity] tests whether milder weights change this picture.

@table: rareclass

The confusion matrix of the random forest ([tab:confusion]) shows the errors that remain even for the strongest model. {{rf_analysis_to_exploits}} of the Analysis flows are predicted as Exploits. DoS has a recall of only {{rf_dos_to_dos}}, with {{rf_dos_to_exploits}} of its flows predicted as Exploits and {{rf_dos_to_fuzzers}} as Fuzzers. Backdoor is the worst case: its recall is {{rf_backdoor_to_backdoor}}, and its flows are spread over Exploits ({{rf_backdoor_to_exploits}}), Fuzzers ({{rf_backdoor_to_fuzzers}}), Generic ({{rf_backdoor_to_generic}}) and Reconnaissance ({{rf_backdoor_to_reconnaissance}}). At the same time, {{rf_fuzzers_to_backdoor}} of the Fuzzers flows are predicted as Backdoor. Since the test set has {{test_count_fuzzers}} Fuzzers flows but only {{test_count_backdoor}} Backdoor flows, this leaves Backdoor with a precision of {{rf_backdoor_precision}} and an F1 of {{pc_rf_backdoor}}, compared with {{rf_backdoor_f1_random}} under the random split. Backdoor traffic in the test period apparently differs from Backdoor traffic in training, and part of the Fuzzers traffic in the test period resembles the Backdoor traffic seen in training. From the data alone we cannot tell whether the attack tools or the labelling changed.

@table: confusion

## 6.4. Context-Group Ablation (H2) {#sec:ablation}

[tab:ablation] removes one context group at a time from gated CAG-IDS. None of the changes matters much. Macro-F1 lies between {{abl_no_c1_macro_mean}} without the structural group and {{abl_full_macro_mean}} with the full context, and every difference is smaller than the seed standard deviation of the model concerned. The mean F1 of the four low-volume classes is {{abl_full_low}} with the full context and {{abl_no_c2_low}} without the behavioural group, a change of {{abl_drop_no_c2}}. Removing the structural group changes it by {{abl_drop_no_c1}} and removing the role group by {{abl_drop_no_c3}}. Removing C2 thus neither hurts the rare classes most nor by more than the seed standard deviation, so H2 is not supported. Reconnaissance, the class that host profiles are most often expected to catch, does not benefit from C2 either (F1 {{abl_full_recon}} with it and {{abl_no_c2_recon}} without). Keeping C1 alone gives {{abl_c1_only_macro_mean}}, as good as the full context. Adding the time of day (C4) does not help ({{abl_with_c4_macro}}), so there is no sign that the model exploits the schedule of the capture.

@table: ablation

## 6.5. What the Gate Learns {#sec:gate}

Across the {{gate_n_test_nodes}} host-windows of the test period, the mean gate value of gated CAG-IDS is {{gate_mean}} (mean ± SD over seeds), so the fused representation draws more on the neighbourhood than on the host's own context. Our expectation that hosts with few flows would lean on their context was not borne out. Hosts with at most ten flows in the window ({{gate_share_low}} of the test host-windows) have a mean gate value of {{gate_low_degree}}, and hosts with at least 100 flows ({{gate_share_high}}) have {{gate_high_degree}}. In {{gate_low_higher_seeds}} of the five seeds it is the hosts with few flows that put more weight on topology, the opposite of what we expected. The rank correlation between a host's flow count and its gate value is negative in {{gate_rho_negative_seeds}} seeds, and its size varies widely ({{gate_rho_list}}; [fig:gate]). The four attackers receive a mean gate value of {{gate_attackers}}, slightly above that of the other hosts ({{gate_others}}). The gate therefore does not settle on one stable rule, which fits its small advantage over concatenation.

@fig: figures/fig5_gate_degree.png | 4.6 | Mean gate value of gated CAG-IDS as a function of the number of flows a host has in the window, for the test host-windows and each of the five seeds. Values above 0.5 put more weight on the neighbourhood embedding than on the context embedding. The shape of the curve changes from seed to seed. {#fig:gate}

## 6.6. Effect of the Split Protocol (RQ2) {#sec:protocol}

[tab:protocol] compares the two protocols for every model. With the same features, models and context, the random split raises macro-F1 for all of them: by {{rf_gap}} for the random forest, {{xgb_gap}} for XGBoost, {{mlp_gap}} for the flow-only MLP and between {{gap_gnn_min}} and {{gap_gnn_max}} for the graph models. The two protocols are scored on different test sets, however. The random test set holds {{rnd_test_attack}} attack flows drawn from the whole capture, whereas the chronological one holds {{n_test_attack}} attack flows from the last {{test_hours}} hours only.

@table: protocol

[tab:sameflow] removes this difference by scoring both kinds of model on the {{same_flow_n}} flows that the two test sets share. The advantage of the random split shrinks but does not go away. It is {{same_rf}} for the random forest and {{same_mlp}} for the flow-only MLP, and the interval lies above zero for every model. Between {{same_share_min}} and {{same_share_max}} of each gap survives when the test flows are identical, so most of it comes from training on traffic from the same minutes and attack sessions as the test flows, and the rest from the composition of the random test set.

@table: sameflow

The effect is not the same for every model. Both CAG-IDS variants gain far more than the others, on their own test sets ({{concat_gap}} and {{cagids_gap}}) and on the shared flows ({{same_concat_delta}} and {{same_cagids_delta}}). This is enough to change the ranking. Under the random split, gated CAG-IDS beats the flow-only MLP by {{boot_rnd_cagids_mlp}} and the centrality-initialised baseline by {{boot_rnd_cagids_b4}}; under the chronological split it loses to both ([tab:significance]). A study that relied on the random split would have concluded that host context improves a graph-based detector, which is the opposite of what the chronological evaluation shows.

## 6.7. Why Endpoint Context Hurts a Flat Classifier {#sec:contexttests}

Adding the context of the two endpoints lowered the forest's macro-F1 ([sec:audit]). Two explanations seemed plausible. The 34 context columns might simply dilute the forest's choice of features, taking split candidates away from the 46 flow features. Or the forest might use the context as a stand-in for how particular hosts behaved during training, which no longer holds in the later test period. [tab:contexttests] separates the two.

@table: contexttests

Dilution explains little. A row-shuffled copy of the context, with the same distribution but no link to the flows or their labels, costs only {{ct_delta_flow_plus_shuffled_context}} macro-F1, against {{ct_delta_flow_plus_context}} for the real context, even though the forest still spends {{ct_chronological_flow_plus_shuffled_context_share}} of its impurity importance on the shuffled columns. Most of the damage comes from the behavioural group. Without C2 the loss shrinks to {{ct_delta_flow_plus_context_without_c2}}, while C2 alone costs {{ct_delta_flow_plus_c2_only}}. The behavioural features change from window to window, and the age feature grows steadily over the capture, so together they can tell the forest roughly when and on which host a flow occurred. That knowledge is useful only if the test traffic comes from the same period as the training traffic.

The random split shows how useful it can be. There, the same context raises the forest's macro-F1 from {{ct_random_flow_flow_only_mean}} to {{ct_random_flow_flow_plus_context_mean}} ({{ct_delta_random_context}}), and Backdoor F1 climbs from {{ct_random_backdoor_flow}} to {{ct_random_backdoor_context}}. A random test flow shares its window with training flows from the same hosts, so its endpoint context is identical to theirs, and the context becomes a key that points to the labels of nearby training flows. The features that lower macro-F1 on later traffic thus raise it by a wide margin when training and test flows are interleaved, which matches the much larger random-split gain of the two CAG-IDS variants ([sec:protocol]).

## 6.8. Sensitivity Analyses {#sec:sensitivity}

[tab:sensitivity] repeats the five neural models with other window lengths and, after the planned runs, with other training settings. The window length matters more than we expected. Compared with the planned 60-s windows, 30-s windows change macro-F1 by {{sens_w30_gain_min}} to {{sens_w30_gain_max}} and 300-s windows by {{sens_w300_gain_min}} to {{sens_w300_gain_max}}, in line with the large effect of the snapshot window reported by Zhao and Hassan [@zhao2026firstprinciples].

@table: sensitivity

Of the training settings, the class weighting matters most. Square-root weights raise macro-F1 by {{sens_sw_gain_min}} to {{sens_sw_gain_max}}, mostly because they cut the false alarms on the rare classes, and the flow-only MLP then reaches {{sens_sw_mlp}}. Focal loss changes macro-F1 by {{sens_focal_gain_min}} to {{sens_focal_gain_max}}, and training for up to 60 epochs with a patience of 12 by {{sens_long_gain_min}} to {{sens_long_gain_max}}. None of these settings brings a neural model close to the random forest: the best score in the whole table is {{sens_best_overall}}, still {{sens_max_gap_to_rf}} below the forest. The flow-only MLP is the best neural model in all {{sens_rows_total}} settings, and no graph model beats it in any of them. The small differences among the graph models are less stable. Gated CAG-IDS stays ahead of the E-GraphSAGE-style baseline in {{sens_rows_cagids_above_b3}} of the {{sens_rows_total}} settings, but the centrality-initialised baseline, which beat it in the planned runs, is ahead in only {{sens_rows_b4_above_cagids}}. The two main conclusions of [sec:main], that no neural model approaches the forest and that message passing does not improve on the flow features, hold in every setting; the order of B4 and gated CAG-IDS does not.

## 6.9. Processing Cost {#sec:cost}

[tab:efficiency] lists the measured costs. Building the window graph and all context features takes {{build_ms}} ms per 60-s window in a single Python process, less than 0.01% of the window length. A forward pass of gated CAG-IDS over one window takes {{infer_ms_cagids}} ms on the GPU, against {{infer_ms_mlp}} ms for the flow-only MLP and {{infer_ms_gatv2}} ms for the GATv2 variant, and a complete training run of gated CAG-IDS takes {{train_s_cagids}} s. On all CPU cores, the random forest trains in {{rf_train_s}} s and XGBoost in {{xgb_train_s}} s, and their batch prediction runs at {{rf_flows_per_second}} and {{xgb_flows_per_second}} flows per second. Every configuration handles a window in a few milliseconds, so processing cost is not what separates these models.

@table: efficiency

# 7. Discussion {#sec:discussion}

## 7.1. What Host Context Can Be Credited With on This Benchmark

On NF-UNSW-NB15-v3, telling attacks apart from benign traffic does not test host context at all. Four addresses produce every attack and nothing else, so any signal tied to hosts can single them out, whether it is an address, a learned memory or a behavioural profile. The context-only probes make the point: without a single flow feature, the forest reaches a binary F1 of {{context_only_rf_binary}} and the neural probe {{m_ctxonly_binary}}. This agrees with earlier warnings about metadata contamination [@dhooge2022contaminating] and design smells in datasets [@flood2024smells]. For host-context methods it matters even more, because describing hosts is exactly what they are for. On this dataset, a binary F1 close to one or a false-positive rate close to zero is no evidence for any detector, and that includes ours.

Attack types are a different story, and there the context did not deliver. Starting the nodes from context rather than from a constant vector helped ({{boot_cagids_b3}}), but the five structural features alone did as well as all seventeen, the behavioural group did not help the rare classes, and in the planned runs the gated architecture did slightly worse than a model in which the context only initialises the message passing, a difference that did not hold across the sensitivity runs. The audit and the probes suggest why. In this closed testbed the same four attackers are active in every period and almost no host meets a new peer, so the behavioural features have little to measure. What they do capture, together with the age of a host, is roughly when and where a flow happened. That knowledge pays off when training and test flows are interleaved, as under the random split, where the context lifted the forest by {{ct_delta_random_context}} and the two CAG-IDS variants gained far more than the other models. On later traffic it does not carry over. Host context may still help on data with many hosts that change their behaviour over time, but this benchmark cannot show it.

## 7.2. The Binary Task Is Saturated; Attack Types Are the Real Problem

Every model we trained, including the probes that see only host context, separates benign from attack traffic almost perfectly. Binary F1, the false-positive rate, AP and weighted F1 therefore cannot rank models on this benchmark, and a study that reported only these metrics on UNSW-NB15-derived data would not be able to see a real difference between detectors. Telling attack types apart is a very different matter. Backdoor, Analysis and DoS reach an F1 of at most {{max_f1_hard_classes}} for any model, and most errors come from a few specific confusions. UNSW-NB15 attack categories look alike at the flow level, and some of the remaining error may come from the labels rather than from the models [@engelen2021troubleshooting, @flood2024smells]. This is where host context would have to prove its value, and on this benchmark it did not.

## 7.3. Why the Neural Models Trail the Forest {#sec:whyneural}

Three things set the neural models back on this benchmark. The first is the class weighting. With inverse-frequency weights, the networks over-predict the rare classes ([tab:rareclass]). Milder, square-root weights reduce this and change macro-F1 by {{sens_sw_gain_min}} to {{sens_sw_gain_max}} ([sec:sensitivity]), but they close only part of the gap to the forest. The second is the graph itself. A 60-s window is a small, dense multigraph in which a host's neighbourhood mixes hundreds of flows of different classes, and averaging their messages blurs the evidence of the one flow being classified. The classifier still sees that flow through its residual input, so message passing adds mostly noise, which fits the loss of the E-GraphSAGE-style baseline against the flow-only MLP. The third is that tree ensembles remain hard to beat on tabular flow features, and the study of Zhao and Hassan likewise found a model without a graph encoder at or near the top [@zhao2026firstprinciples]. Longer training helped some models slightly but did not change their ranking. None of this rules out that a carefully tuned GNN would do better, but under one shared protocol the graph models gave no advantage.

## 7.4. Implications for Evaluating Graph-Based Detectors

Our results suggest four simple practices, none of them expensive. First, before giving a host-level representation credit for a gain, check how much of the label host identity alone explains; a majority-class lookup takes seconds. Second, evaluate on later traffic, and when a random split is reported, also score it on flows from a later period. Here, the random split raised macro-F1 by {{gap_flat_min}} to {{gap_flat_max}} for the flat baselines and by up to {{gap_gnn_max}} for the graph models, most of the gain survived on identical test flows, and it reversed the conclusion about host context. Host features deserve particular care, because under a random split they turn into a key to the labels of training flows from the same window. Third, report flat baselines, a flow-only neural model and simple probes next to the graph models, because a model without a graph encoder may well be the strongest one, as it was here and in the study of Zhao and Hassan [@zhao2026firstprinciples]. Fourth, report macro-F1 and per-class precision and recall instead of metrics that are dominated by the benign class; the over-prediction of rare classes that we observed is invisible in binary F1 and AP.

## 7.5. Deployment Considerations {#sec:deployment}

A window-causal detector adds up to Δ (60 s) of delay. That is acceptable for most attack classes, but not for very fast ones. The behavioural baselines learn from everything a host has done before, attacks included, so a host that keeps attacking slowly redefines its own normal. In our experiments the history simply carries on through the test period, but a real deployment would need to freeze, or carefully update, the baselines of hosts that have already been flagged. New hosts get neutral z-scores, which is a reasonable default but also gives an opening to an attacker who keeps changing addresses. An attacker who knows the features could also stay within its own baseline and spread its activity over many windows, so context makes evasion harder but does not prevent it. The processing cost is not an obstacle: building the inputs and running gated CAG-IDS takes a few milliseconds per 60-s window ([sec:cost]). Finally, keeping per-host profiles raises privacy questions, and how long the history is kept should follow local policy.

# 8. Limitations and Threats to Validity {#sec:limitations}

***Internal validity.*** All variants share one implementation and one set of hyperparameters, fixed before the runs. This removes implementation differences, but none of the models is tuned to its best, and the neural models in particular are sensitive to how the classes are weighted ([sec:sensitivity]). The sensitivity runs on class weighting, loss and training length were added after the planned runs and should be read as exploratory; we did not select any headline result from them. The variants differ in size, and B3 and B4 lack the direct context path, which the C1 + gate control only partly repairs. B3 and B4 follow the ideas of [@lo2022egraphsage] and [@termos2025centrality] rather than every detail, so our comparison is between ways of initialising nodes, not between the published systems. The flat baselines train on a 400,000-flow subsample while the neural models see all training flows, and GPU scatter operations add a small amount of randomness beyond the seed.

***Construct validity.*** Macro-F1 gives the {{test_count_worms}} Worms test flows the same weight as the {{test_count_exploits}} Exploits flows, so a few Worms predictions can shift it noticeably. Attack categories overlap at the flow level, and the labels depend on direction, since replies from victims are labelled benign ([sec:audit]). Categorical NetFlow codes and DNS_QUERY_ID enter the models as standardised numbers ([sec:features]); this is a crude encoding for the codes, and the identifier could produce spurious patterns. The behavioural features are only updated in windows where a host is active, so a host that has been quiet for a while is compared with its last active windows and not with the time that has passed. The gate analysis uses the gate value averaged over the 128 dimensions of each host, which hides any structure across dimensions.

***External validity.*** A single testbed dataset, with four attackers active in every split, cannot show that our results carry over to real networks, where attackers are new, malicious flows are rare and benign hosts change roles. The chronological split protects against temporal leakage but not against host overlap, because there is no unseen attacker to hold out. Cross-dataset studies also suggest caution [@layeghy2023crossdomain]. We had planned NF-ToN-IoT-v3 [@luay2025temporal, @booij2022toniot] as a second dataset but did not evaluate it.

***Conclusion validity.*** Five seeds and one chronological split give only a modest basis for inference. The block bootstrap accounts for variation in the test period, but it is no substitute for an independent capture. The planned comparisons are not corrected for multiple testing, and the same-flow comparison of the two split protocols rests on the {{same_flow_n}} flows that the two test sets share.

# 9. Conclusion and Future Work {#sec:conclusion}

We set out to find whether explicit, causally computed host context helps a graph-based flow classifier. CAG-IDS combines structural, behavioural and role context from the current and earlier windows with topology through a learned gate, at a cost of a few milliseconds per window. On NF-UNSW-NB15-v3 the context did not help, and our experiments show why. First, host identity fully determines whether a flow is benign or an attack, and TTL almost does, so binary metrics cannot evaluate host-context methods on this benchmark. Second, telling the attack types apart remains hard, and the flat random forest ({{m_rf_macro_mean}}) beats every neural model; the best neural model is a flow-only MLP ({{m_mlp_macro_mean}}). Third, host context helped a graph model only relative to constant node features. It beat the E-GraphSAGE-style baseline by {{boot_cagids_b3_delta}} but not the centrality-initialised one, whose small lead changed sign with the window length and the training settings; the behavioural and role groups added nothing to the structural features, the gate was not better than concatenation, and the behavioural group did not help the rare classes. Fourth, a random flow split raised macro-F1 for every model, kept most of that gain on identical test flows, favoured the context-aware models most and would have reversed the conclusion about host context, because under that split the context of a test flow points to the labels of training flows from the same window.

The broader lesson is about method. Before giving host context, or any host-level representation, credit for a gain on a benchmark, it is worth checking how much of that gain host identity and temporal proximity alone would deliver. Several next steps follow from our limitations. We would like to evaluate on datasets with more attacking hosts, such as NF-ToN-IoT-v3 and NF-CSE-CIC-IDS2018-v3 [@luay2025temporal], which may allow splits in which the test attackers never appear in training. Other directions are class weighting designed for extreme imbalance, a continuous-time model with per-host memory [@rossi2020tgn] to remove the window delay, behavioural baselines that are robust to traffic already judged malicious, and explanations of each decision in terms of the context features behind it.

# Data and Code Availability

NF-UNSW-NB15-v3 is available from the University of Queensland data repository under the terms set by its creators [@luay2025temporal]. The implementation, the configuration files and the scripts that produce every results table, every figure and every reported number are available from the authors on request. The script scripts/run_all.py runs the full experiment suite, scripts/analyze.py regenerates the results tables, data figures and quoted values, and paper/build_docx.py assembles this manuscript from them.

# References

@references
