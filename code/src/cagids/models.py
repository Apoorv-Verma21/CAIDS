import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(messages, index, n_nodes):
    out = torch.zeros(n_nodes, messages.size(1), device=messages.device, dtype=messages.dtype)
    out.index_add_(0, index, messages)
    counts = torch.zeros(n_nodes, device=messages.device, dtype=messages.dtype)
    counts.index_add_(0, index, torch.ones(index.size(0), device=messages.device, dtype=messages.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    return out / counts


def scatter_max(messages, index, n_nodes):
    # nodes that receive no message keep zeros
    out = torch.zeros(n_nodes, messages.size(1), device=messages.device, dtype=messages.dtype)
    idx = index.unsqueeze(1).expand_as(messages)
    return out.scatter_reduce(0, idx, messages, reduce="amax", include_self=False)


def aggregate(messages, index, n_nodes, how="mean"):
    if how == "max":
        return scatter_max(messages, index, n_nodes)
    return scatter_mean(messages, index, n_nodes)


def _both_directions(edge_src, edge_dst, edge_attr):
    
    senders = torch.cat([edge_src, edge_dst], dim=0)
    receivers = torch.cat([edge_dst, edge_src], dim=0)
    attr = torch.cat([edge_attr, edge_attr], dim=0)
    return senders, receivers, attr


class EdgeAwareSAGELayer(nn.Module):
    """Edge-aware GraphSAGE layer. By default every flow sends the same kind of
    message to both endpoints and the messages are averaged. aggregator="max"
    takes the element-wise maximum instead. direction_aware=True gives flows a
    host receives and flows it sends separate message weights and separate
    aggregates, so the update can tell the two apart."""

    def __init__(self, in_dim, edge_dim, out_dim, dropout=0.2, aggregator="mean", direction_aware=False):
        super().__init__()
        self.aggregator = aggregator
        self.direction_aware = direction_aware
        self.message = nn.Linear(in_dim + edge_dim, out_dim)
        if direction_aware:
            self.message_out = nn.Linear(in_dim + edge_dim, out_dim)
            self.update = nn.Linear(in_dim + 2 * out_dim, out_dim)
        else:
            self.update = nn.Linear(in_dim + out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_src, edge_dst, edge_attr):
        n_nodes = h.size(0)
        if self.direction_aware:
            # flows the host receives: message built from the sender
            inbound = self.message(torch.cat([h[edge_src], edge_attr], dim=1))
            agg_in = aggregate(inbound, edge_dst, n_nodes, self.aggregator)
            # flows the host sends: message built from the destination
            outbound = self.message_out(torch.cat([h[edge_dst], edge_attr], dim=1))
            agg_out = aggregate(outbound, edge_src, n_nodes, self.aggregator)
            out = self.update(torch.cat([h, agg_in, agg_out], dim=1))
            return self.dropout(F.relu(out))
        senders, receivers, attr = _both_directions(edge_src, edge_dst, edge_attr)
        messages = self.message(torch.cat([h[senders], attr], dim=1))
        aggregated = aggregate(messages, receivers, n_nodes, self.aggregator)
        out = self.update(torch.cat([h, aggregated], dim=1))
        return self.dropout(F.relu(out))


class EdgeAwareGATv2Layer(nn.Module):
    

    def __init__(self, in_dim, edge_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.src_proj = nn.Linear(in_dim, out_dim)
        self.dst_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.update = nn.Linear(in_dim + out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_src, edge_dst, edge_attr):
        senders, receivers, attr = _both_directions(edge_src, edge_dst, edge_attr)
        n_nodes = h.size(0)
        x_src = self.src_proj(h)[senders]
        x_dst = self.dst_proj(h)[receivers]
        x_edge = self.edge_proj(attr)

        values = (x_src + x_edge).view(-1, self.heads, self.head_dim)
        hidden = F.leaky_relu(x_src + x_dst + x_edge, 0.2).view(-1, self.heads, self.head_dim)
        logits = (hidden * self.att).sum(dim=-1)

        index = receivers.unsqueeze(1).expand(-1, self.heads)
        peak = torch.full((n_nodes, self.heads), float("-inf"), device=h.device, dtype=logits.dtype)
        peak = peak.scatter_reduce(0, index, logits, reduce="amax", include_self=True)
        weights = torch.exp(logits - peak[receivers])
        total = torch.zeros(n_nodes, self.heads, device=h.device, dtype=logits.dtype)
        total.index_add_(0, receivers, weights)
        alpha = self.dropout(weights / total[receivers].clamp(min=1e-16))

        aggregated = torch.zeros(n_nodes, self.heads, self.head_dim, device=h.device, dtype=h.dtype)
        aggregated.index_add_(0, receivers, alpha.unsqueeze(-1) * values)
        aggregated = aggregated.view(n_nodes, -1)

        out = self.update(torch.cat([h, aggregated], dim=1))
        return self.dropout(F.relu(out))


class ContextEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, context):
        return self.net(context)


class CAGIDS(nn.Module):
    

    def __init__(self, edge_dim, context_dim, n_classes, cfg):
        super().__init__()
        model_cfg = cfg.model
        hidden = int(model_cfg.get("hidden_dim", 128))
        ctx_dim = int(model_cfg.get("context_dim", 64))
        layers = int(model_cfg.get("layers", 2))
        dropout = float(model_cfg.get("dropout", 0.2))
        backbone = model_cfg.get("backbone", "sage")
        heads = int(model_cfg.get("heads", 4))
        aggregator = model_cfg.get("aggregator", "mean")
        direction_aware = bool(model_cfg.get("direction_aware", False))

        self.node_init = model_cfg.get("node_init", "context")
        self.fusion = model_cfg.get("fusion", "gate")
        self.use_edge_residual = bool(model_cfg.get("edge_residual", True))
        self.structural_slice = int(model_cfg.get("structural_dim", 5))

        if self.node_init == "none":
            self.encoder = None
            self.gnn = nn.ModuleList()
            self.gate = None
            self.context_project = None
            self.classifier = nn.Sequential(
                nn.Linear(edge_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            return

        if self.node_init == "ones":
            init_dim = hidden
            self.encoder = None
        elif self.node_init == "centrality":
            init_dim = ctx_dim
            self.encoder = ContextEncoder(self.structural_slice, hidden, ctx_dim, dropout)
        else:
            init_dim = ctx_dim
            self.encoder = ContextEncoder(context_dim, hidden, ctx_dim, dropout)

        self.init_dim = init_dim
        self.gnn = nn.ModuleList()
        dim = init_dim
        for _ in range(layers):
            if backbone == "gatv2":
                self.gnn.append(EdgeAwareGATv2Layer(dim, edge_dim, hidden, heads, dropout))
            else:
                self.gnn.append(EdgeAwareSAGELayer(dim, edge_dim, hidden, dropout, aggregator, direction_aware))
            dim = hidden
        self.out_dim = dim

        has_context = self.encoder is not None and self.node_init == "context"
        if self.fusion == "gate" and has_context and layers > 0:
            self.gate = nn.Linear(self.out_dim + ctx_dim, self.out_dim)
            self.context_project = nn.Linear(ctx_dim, self.out_dim)
            fused_dim = self.out_dim
        elif self.fusion == "concat" and has_context and layers > 0:
            self.gate = None
            self.context_project = None
            fused_dim = self.out_dim + ctx_dim
        else:
            self.gate = None
            self.context_project = None
            fused_dim = self.out_dim

        classifier_in = 2 * fused_dim + (edge_dim if self.use_edge_residual else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def initial_embedding(self, context):
        if self.node_init == "ones":
            return torch.ones(context.size(0), self.init_dim, device=context.device)
        if self.node_init == "centrality":
            return self.encoder(context[:, : self.structural_slice])
        return self.encoder(context)

    def forward(self, context, edge_src, edge_dst, edge_attr, return_gate=False):
        if self.node_init == "none":
            logits = self.classifier(edge_attr)
            return (logits, None) if return_gate else logits

        encoded = self.initial_embedding(context)
        h = encoded
        for layer in self.gnn:
            h = layer(h, edge_src, edge_dst, edge_attr)

        gate = None
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(torch.cat([h, encoded], dim=1)))
            z = gate * h + (1.0 - gate) * self.context_project(encoded)
        elif self.fusion == "concat" and self.encoder is not None and len(self.gnn) > 0 \
                and self.node_init == "context":
            z = torch.cat([h, encoded], dim=1)
        else:
            z = h

        parts = [z[edge_src], z[edge_dst]]
        if self.use_edge_residual:
            parts.append(edge_attr)
        logits = self.classifier(torch.cat(parts, dim=1))
        return (logits, gate) if return_gate else logits


EXPERIMENT_MODELS = {
    "flow_mlp": {"node_init": "none", "fusion": "none"},
    "egraphsage": {"node_init": "ones", "fusion": "none"},
    "centrality_egraphsage": {"node_init": "centrality", "fusion": "none"},
    "cagids_concat": {"node_init": "context", "fusion": "concat"},
    "cagids": {"node_init": "context", "fusion": "gate"},
    "cagids_gatv2": {"node_init": "context", "fusion": "gate", "backbone": "gatv2"},
    "context_only": {"node_init": "context", "fusion": "none", "layers": 0, "edge_residual": False},
}


def build_model(edge_dim, context_dim, n_classes, cfg):
    name = cfg.model.get("name", "cagids")
    if name in EXPERIMENT_MODELS:
        merged = dict(cfg["model"])
        merged.update(EXPERIMENT_MODELS[name])
        cfg = type(cfg)({**cfg, "model": merged})
    return CAGIDS(edge_dim, context_dim, n_classes, cfg)
