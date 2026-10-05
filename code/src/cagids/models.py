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


class EdgeAwareSAGELayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, dropout=0.2):
        super().__init__()
        self.message = nn.Linear(in_dim + edge_dim, out_dim)
        self.update = nn.Linear(in_dim + out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_src, edge_dst, edge_attr):
        msg_forward = self.message(torch.cat([h[edge_src], edge_attr], dim=1))
        msg_backward = self.message(torch.cat([h[edge_dst], edge_attr], dim=1))

        index = torch.cat([edge_dst, edge_src], dim=0)
        messages = torch.cat([msg_forward, msg_backward], dim=0)
        aggregated = scatter_mean(messages, index, h.size(0))

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

        self.node_init = model_cfg.get("node_init", "context")
        self.fusion = model_cfg.get("fusion", "gate")
        self.structural_slice = int(model_cfg.get("structural_dim", 5))

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
            self.gnn.append(EdgeAwareSAGELayer(dim, edge_dim, hidden, dropout))
            dim = hidden
        self.out_dim = dim

        if self.fusion == "gate" and self.encoder is not None:
            self.gate = nn.Linear(self.out_dim + ctx_dim, self.out_dim)
            self.context_project = nn.Linear(ctx_dim, self.out_dim)
            fused_dim = self.out_dim
        elif self.fusion == "concat" and self.encoder is not None:
            self.gate = None
            self.context_project = None
            fused_dim = self.out_dim + ctx_dim
        else:
            self.gate = None
            self.context_project = None
            fused_dim = self.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(2 * fused_dim + edge_dim, hidden),
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

    def forward(self, context, edge_src, edge_dst, edge_attr):
        encoded = self.initial_embedding(context)
        h = encoded
        for layer in self.gnn:
            h = layer(h, edge_src, edge_dst, edge_attr)

        if self.gate is not None:
            g = torch.sigmoid(self.gate(torch.cat([h, encoded], dim=1)))
            z = g * h + (1.0 - g) * self.context_project(encoded)
        elif self.fusion == "concat" and self.encoder is not None:
            z = torch.cat([h, encoded], dim=1)
        else:
            z = h

        flow_repr = torch.cat([z[edge_src], z[edge_dst], edge_attr], dim=1)
        return self.classifier(flow_repr)


def build_model(edge_dim, context_dim, n_classes, cfg):
    name = cfg.model.get("name", "cagids")
    overrides = {
        "egraphsage": {"node_init": "ones", "fusion": "none"},
        "centrality_egraphsage": {"node_init": "centrality", "fusion": "none"},
        "cagids": {"node_init": "context", "fusion": "gate"},
        "cagids_concat": {"node_init": "context", "fusion": "concat"},
    }
    if name in overrides:
        merged = dict(cfg["model"])
        merged.update(overrides[name])
        cfg = type(cfg)({**cfg, "model": merged})
    return CAGIDS(edge_dim, context_dim, n_classes, cfg)
