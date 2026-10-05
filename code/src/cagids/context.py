import ipaddress

import numpy as np
from scipy import sparse


STRUCTURAL_NAMES = [
    "in_degree",
    "out_degree",
    "unique_peers",
    "pagerank",
    "clustering",
]

BEHAVIOURAL_NAMES = [
    "z_flows",
    "z_bytes",
    "z_unique_ports",
    "z_unique_peers",
    "peer_novelty",
    "log_age_windows",
    "is_first_seen",
]

ROLE_NAMES = [
    "is_private",
    "inbound_ratio",
    "server_port_share",
    "dst_port_entropy",
    "log_total_flows",
]

TIME_NAMES = ["tod_sin", "tod_cos"]

WELL_KNOWN_PORTS = {20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123, 137, 138, 139,
                    143, 161, 389, 443, 445, 465, 514, 587, 631, 993, 995, 1433, 1521,
                    3306, 3389, 5432, 5900, 8080, 8443}


def context_feature_names(cfg):
    names = []
    if cfg.context.get("use_structural", True):
        names += STRUCTURAL_NAMES
    if cfg.context.get("use_behavioural", True):
        names += BEHAVIOURAL_NAMES
    if cfg.context.get("use_role", True):
        names += ROLE_NAMES
    if cfg.context.get("use_time", False):
        names += TIME_NAMES
    return names


def _is_private(ip_string):
    try:
        return 1.0 if ipaddress.ip_address(ip_string).is_private else 0.0
    except ValueError:
        return 0.0


def _pagerank(src, dst, n_nodes, iters, damping):
    if n_nodes == 0:
        return np.zeros(0, dtype="float32")
    data = np.ones(len(src), dtype="float32")
    adj = sparse.coo_matrix((data, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = out_deg == 0
    out_deg[dangling] = 1.0
    inv = sparse.diags(1.0 / out_deg)
    transition = (inv @ adj).T.tocsr()

    rank = np.full(n_nodes, 1.0 / n_nodes, dtype="float64")
    for _ in range(iters):
        leaked = damping * rank[dangling].sum() / n_nodes
        rank = (1.0 - damping) / n_nodes + damping * (transition @ rank) + leaked
        total = rank.sum()
        if total > 0:
            rank = rank / total
    return rank.astype("float32")


def _clustering(src, dst, n_nodes):
    if n_nodes == 0:
        return np.zeros(0, dtype="float32")
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    if len(rows) == 0:
        return np.zeros(n_nodes, dtype="float32")
    data = np.ones(len(rows), dtype="float32")
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()

    degree = np.asarray(adj.sum(axis=1)).ravel()
    triangles = np.asarray((adj @ adj).multiply(adj).sum(axis=1)).ravel() / 2.0
    possible = degree * (degree - 1) / 2.0
    out = np.zeros(n_nodes, dtype="float32")
    mask = possible > 0
    out[mask] = (triangles[mask] / possible[mask]).astype("float32")
    return out


class HostHistory:
    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.mean = {}
        self.var = {}
        self.peers = {}
        self.first_window = {}

    def stats(self, host, values):
        mean = self.mean.get(host)
        if mean is None:
            return np.zeros(len(values), dtype="float32"), True
        var = self.var[host]
        std = np.sqrt(np.maximum(var, 1e-6))
        z = (values - mean) / std
        return np.clip(z, -10.0, 10.0).astype("float32"), False

    def novelty(self, host, current_peers):
        seen = self.peers.get(host)
        if not current_peers:
            return 0.0
        if seen is None:
            return 1.0
        new = len(current_peers - seen)
        return float(new) / float(len(current_peers))

    def age(self, host, window_index):
        first = self.first_window.get(host)
        if first is None:
            return 0.0
        return float(window_index - first)

    def update(self, host, values, current_peers, window_index):
        a = self.alpha
        mean = self.mean.get(host)
        if mean is None:
            self.mean[host] = values.astype("float64").copy()
            self.var[host] = np.ones(len(values), dtype="float64")
            self.first_window[host] = window_index
        else:
            delta = values - mean
            self.mean[host] = mean + a * delta
            self.var[host] = (1 - a) * (self.var[host] + a * delta * delta)
        if host in self.peers:
            self.peers[host].update(current_peers)
        else:
            self.peers[host] = set(current_peers)


class ContextExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = HostHistory(cfg.context.get("ewma_alpha", 0.3))
        self.names = context_feature_names(cfg)
        self.use_structural = cfg.context.get("use_structural", True)
        self.use_behavioural = cfg.context.get("use_behavioural", True)
        self.use_role = cfg.context.get("use_role", True)
        self.use_time = cfg.context.get("use_time", False)
        self._private_cache = {}

    def _private(self, ip_string):
        value = self._private_cache.get(ip_string)
        if value is None:
            value = _is_private(ip_string)
            self._private_cache[ip_string] = value
        return value

    def process_window(self, window_index, nodes, src_local, dst_local,
                       src_port, dst_port, total_bytes, start_ms):
        n_nodes = len(nodes)
        blocks = []

        in_deg = np.bincount(dst_local, minlength=n_nodes).astype("float32")
        out_deg = np.bincount(src_local, minlength=n_nodes).astype("float32")

        peers = [set() for _ in range(n_nodes)]
        dst_ports_per_host = [[] for _ in range(n_nodes)]
        bytes_per_host = np.zeros(n_nodes, dtype="float64")

        for i in range(len(src_local)):
            s = src_local[i]
            d = dst_local[i]
            peers[s].add(d)
            peers[d].add(s)
            dst_ports_per_host[s].append(int(dst_port[i]))
            bytes_per_host[s] += total_bytes[i]
            bytes_per_host[d] += total_bytes[i]

        unique_peers = np.array([len(p) for p in peers], dtype="float32")

        if self.use_structural:
            pagerank = _pagerank(
                src_local, dst_local, n_nodes,
                int(self.cfg.context.get("pagerank_iters", 30)),
                float(self.cfg.context.get("pagerank_damping", 0.85)),
            )
            clustering = _clustering(src_local, dst_local, n_nodes)
            blocks.append(np.column_stack([
                np.log1p(in_deg),
                np.log1p(out_deg),
                np.log1p(unique_peers),
                pagerank * n_nodes,
                clustering,
            ]).astype("float32"))

        n_flows = in_deg + out_deg
        unique_ports = np.array(
            [len(set(ports)) for ports in dst_ports_per_host], dtype="float32"
        )

        if self.use_behavioural:
            behaviour = np.zeros((n_nodes, len(BEHAVIOURAL_NAMES)), dtype="float32")
            for idx in range(n_nodes):
                host = nodes[idx]
                observation = np.array([
                    np.log1p(n_flows[idx]),
                    np.log1p(bytes_per_host[idx]),
                    np.log1p(unique_ports[idx]),
                    np.log1p(unique_peers[idx]),
                ], dtype="float64")
                z, cold = self.history.stats(host, observation)
                peer_names = {nodes[p] for p in peers[idx]}
                behaviour[idx, 0:4] = z
                behaviour[idx, 4] = self.history.novelty(host, peer_names)
                behaviour[idx, 5] = np.log1p(self.history.age(host, window_index))
                behaviour[idx, 6] = 1.0 if cold else 0.0
            blocks.append(behaviour)

        if self.use_role:
            role = np.zeros((n_nodes, len(ROLE_NAMES)), dtype="float32")
            for idx in range(n_nodes):
                total = n_flows[idx]
                inbound = in_deg[idx] / total if total > 0 else 0.0
                ports = dst_ports_per_host[idx]
                if ports:
                    server_share = sum(1 for p in ports if p in WELL_KNOWN_PORTS) / len(ports)
                    counts = np.unique(np.array(ports), return_counts=True)[1].astype("float64")
                    probs = counts / counts.sum()
                    entropy = float(-(probs * np.log2(probs)).sum())
                else:
                    server_share = 0.0
                    entropy = 0.0
                role[idx] = [
                    self._private(nodes[idx]),
                    inbound,
                    server_share,
                    entropy,
                    np.log1p(total),
                ]
            blocks.append(role)

        if self.use_time:
            seconds = (float(start_ms) / 1000.0) % 86400.0
            angle = 2.0 * np.pi * seconds / 86400.0
            time_block = np.tile(
                np.array([np.sin(angle), np.cos(angle)], dtype="float32"), (n_nodes, 1)
            )
            blocks.append(time_block)

        if self.use_behavioural:
            for idx in range(n_nodes):
                observation = np.array([
                    np.log1p(n_flows[idx]),
                    np.log1p(bytes_per_host[idx]),
                    np.log1p(unique_ports[idx]),
                    np.log1p(unique_peers[idx]),
                ], dtype="float64")
                peer_names = {nodes[p] for p in peers[idx]}
                self.history.update(nodes[idx], observation, peer_names, window_index)

        if not blocks:
            return np.ones((n_nodes, 1), dtype="float32")
        return np.concatenate(blocks, axis=1).astype("float32")
