import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import compute_metrics, macro_f1_present

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
IGNORE = -100


def resolve_device(setting="auto"):
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def windows_for_split(graphs, edge_split, split_name):
    """Windows that contain at least one flow of the given split."""
    code = SPLIT_CODES[split_name]
    hit = edge_split == code
    if not hit.any():
        return []
    counts = np.add.reduceat(hit.astype("int64"), graphs.edge_offsets[:-1])
    empty = graphs.edge_offsets[:-1] == graphs.edge_offsets[1:]
    counts[empty] = 0
    return [int(w) for w in np.flatnonzero(counts > 0)]


def collate_windows(graphs, window_indices, edge_features, labels, edge_split, split_code, device):
    src_parts, dst_parts, context_parts, attr_parts, label_parts, mask_parts = [], [], [], [], [], []
    edge_ids, node_ids = [], []
    node_offset = 0

    for w in window_indices:
        e0, e1 = int(graphs.edge_offsets[w]), int(graphs.edge_offsets[w + 1])
        n0, n1 = int(graphs.node_offsets[w]), int(graphs.node_offsets[w + 1])

        src_parts.append(graphs.edge_src[e0:e1].astype("int64") + node_offset)
        dst_parts.append(graphs.edge_dst[e0:e1].astype("int64") + node_offset)
        context_parts.append(graphs.context[n0:n1])
        attr_parts.append(edge_features[e0:e1])
        label_parts.append(labels[e0:e1])
        mask_parts.append(edge_split[e0:e1] == split_code)
        edge_ids.append(np.arange(e0, e1, dtype="int64"))
        node_ids.append(np.arange(n0, n1, dtype="int64"))
        node_offset += n1 - n0

    mask = np.concatenate(mask_parts)
    target = np.concatenate(label_parts).astype("int64")
    masked_target = np.where(mask, target, IGNORE)
    return {
        "src": torch.from_numpy(np.concatenate(src_parts)).to(device),
        "dst": torch.from_numpy(np.concatenate(dst_parts)).to(device),
        "context": torch.from_numpy(np.concatenate(context_parts, axis=0)).to(device),
        "edge_attr": torch.from_numpy(np.concatenate(attr_parts, axis=0)).to(device),
        "labels": torch.from_numpy(masked_target).to(device),
        "mask": mask,
        "edge_ids": np.concatenate(edge_ids),
        "node_ids": np.concatenate(node_ids),
        "n_nodes": node_offset,
    }


def class_weights(labels, n_classes, scheme="inverse"):
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    if scheme == "none":
        return np.ones(n_classes, dtype="float32")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    if scheme == "sqrt_inverse":
        weights = np.sqrt(weights)
    return weights.astype("float32")


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        keep = target != IGNORE
        logits, target = logits[keep], target[keep]
        if target.numel() == 0:
            return logits.sum() * 0.0
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            w = self.weight[target]
            return (w * loss).sum() / w.sum()
        return loss.mean()


def run_epoch(model, graphs, window_indices, edge_features, labels, edge_split, split_name,
              device, windows_per_batch, optimizer=None, criterion=None, shuffle=False,
              collect_gate=False):
    training = optimizer is not None
    model.train(training)
    split_code = SPLIT_CODES[split_name]

    order = np.array(window_indices)
    if shuffle:
        order = order[np.random.permutation(len(order))]

    total_loss = 0.0
    total_edges = 0
    all_pred, all_true, all_prob, all_logit, all_ids = [], [], [], [], []
    gate_records, gate_nodes = [], []

    for start in range(0, len(order), windows_per_batch):
        batch_windows = order[start:start + windows_per_batch]
        batch = collate_windows(graphs, batch_windows, edge_features, labels, edge_split, split_code, device)
        n = int(batch["mask"].sum())
        if n == 0:
            continue

        with torch.set_grad_enabled(training):
            if collect_gate:
                logits, gate = model(batch["context"], batch["src"], batch["dst"], batch["edge_attr"],
                                     return_gate=True)
            else:
                logits = model(batch["context"], batch["src"], batch["dst"], batch["edge_attr"])
                gate = None
            loss = criterion(logits, batch["labels"])

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += float(loss.item()) * n
        total_edges += n

        if not training:
            mask_t = torch.from_numpy(batch["mask"]).to(device)
            probs = torch.softmax(logits[mask_t], dim=1).detach().cpu().numpy()
            all_prob.append(probs.astype("float32"))
            all_logit.append(logits[mask_t].detach().cpu().numpy().astype("float32"))
            all_pred.append(probs.argmax(axis=1))
            all_true.append(batch["labels"][mask_t].detach().cpu().numpy())
            all_ids.append(batch["edge_ids"][batch["mask"]])
            if gate is not None:
                gate_records.append(gate.mean(dim=1).detach().cpu().numpy())
                gate_nodes.append(batch["node_ids"])

    result = {"loss": total_loss / max(total_edges, 1)}
    if not training and all_true:
        result["y_true"] = np.concatenate(all_true)
        result["y_pred"] = np.concatenate(all_pred)
        result["y_prob"] = np.concatenate(all_prob, axis=0)
        result["y_logit"] = np.concatenate(all_logit, axis=0)
        result["edge_ids"] = np.concatenate(all_ids)
        if gate_records:
            result["gate_mean"] = np.concatenate(gate_records)
            result["gate_nodes"] = np.concatenate(gate_nodes)
    return result


def measure_inference(model, graphs, window_indices, edge_features, device, max_windows=200):
    """Wall-clock forward time for one window at a time, as a deployed detector
    would see it. Context extraction is timed separately in build_graphs."""
    model.eval()
    chosen = list(window_indices)[:max_windows]
    times, flows = [], []
    dummy_split = np.zeros(len(edge_features), dtype="int8")
    labels = np.zeros(len(edge_features), dtype="int64")
    with torch.no_grad():
        for i, w in enumerate(chosen):
            batch = collate_windows(graphs, [w], edge_features, labels, dummy_split, 0, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batch["context"], batch["src"], batch["dst"], batch["edge_attr"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i >= 5:  # skip warm-up
                times.append(elapsed)
                flows.append(len(batch["edge_ids"]))
    if not times:
        return {}
    times = np.array(times)
    flows = np.array(flows)
    return {
        "ms_per_window_mean": float(times.mean() * 1000),
        "ms_per_window_p95": float(np.percentile(times, 95) * 1000),
        "flows_per_second": float(flows.sum() / times.sum()),
        "windows_timed": int(len(times)),
    }


def build_criterion(cfg, weights, device):
    loss_name = cfg.train.get("loss", "ce")
    weight_t = torch.from_numpy(weights).to(device)
    if loss_name == "focal":
        return FocalLoss(weight=weight_t, gamma=float(cfg.train.get("focal_gamma", 2.0)))
    return nn.CrossEntropyLoss(weight=weight_t, ignore_index=IGNORE)


def train_model(model, graphs, edge_split, edge_features, labels, class_names, cfg, verbose=True):
    device = resolve_device(cfg.train.get("device", "auto"))
    model = model.to(device)
    n_classes = len(class_names)

    train_windows = windows_for_split(graphs, edge_split, "train")
    val_windows = windows_for_split(graphs, edge_split, "val")
    test_windows = windows_for_split(graphs, edge_split, "test")
    if not train_windows or not val_windows or not test_windows:
        raise ValueError("train, val and test splits must all be non-empty")

    train_labels = labels[edge_split == SPLIT_CODES["train"]]
    weights = class_weights(train_labels, n_classes, cfg.train.get("class_weighting", "inverse"))
    criterion = build_criterion(cfg, weights, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.get("lr", 0.003)),
        weight_decay=float(cfg.train.get("weight_decay", 1e-5)),
    )

    windows_per_batch = int(cfg.train.get("windows_per_batch", 8))
    patience = int(cfg.train.get("patience", 6))
    # Early stopping can leave out classes that the validation set holds only a
    # handful of flows of; the full validation macro-F1 is still recorded.
    exclude = list(cfg.train.get("early_stop_exclude") or [])
    unknown = [c for c in exclude if c not in class_names]
    if unknown:
        raise ValueError("unknown classes for early stopping: {}".format(", ".join(unknown)))
    es_keep = [i for i, c in enumerate(class_names) if c not in exclude] if exclude else None
    best_full = -1.0
    best_score = -1.0
    best_state = None
    best_epoch = 0
    waited = 0
    history = []
    train_start = time.perf_counter()

    for epoch in range(1, int(cfg.train.get("epochs", 30)) + 1):
        t0 = time.perf_counter()
        train_out = run_epoch(model, graphs, train_windows, edge_features, labels, edge_split, "train",
                              device, windows_per_batch, optimizer, criterion, shuffle=True)
        val_out = run_epoch(model, graphs, val_windows, edge_features, labels, edge_split, "val",
                            device, windows_per_batch, None, criterion)
        if "y_true" not in val_out:
            continue
        full = macro_f1_present(val_out["y_true"], val_out["y_pred"], n_classes)
        score = full if es_keep is None else macro_f1_present(val_out["y_true"], val_out["y_pred"], n_classes, es_keep)
        history.append({
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "val_macro_f1": full,
            "seconds": time.perf_counter() - t0,
        })
        if es_keep is not None:
            history[-1]["val_early_stop_score"] = score

        if verbose:
            print("epoch {:>3} | train loss {:.4f} | val loss {:.4f} | val macro F1 {:.4f} | {:.1f}s".format(
                epoch, train_out["loss"], val_out["loss"], score, history[-1]["seconds"]), flush=True)

        if score > best_score + 1e-5:
            best_score = score
            best_full = full
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                if verbose:
                    print("early stopping at epoch {}".format(epoch), flush=True)
                break

    train_seconds = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    collect_gate = getattr(model, "gate", None) is not None
    test_out = run_epoch(model, graphs, test_windows, edge_features, labels, edge_split, "test",
                         device, windows_per_batch, None, criterion, collect_gate=collect_gate)
    test_metrics = compute_metrics(test_out["y_true"], test_out["y_pred"], class_names, test_out["y_prob"])
    test_metrics["best_val_macro_f1"] = best_full
    if es_keep is not None:
        test_metrics["early_stop_exclude"] = exclude
        test_metrics["best_val_early_stop_score"] = best_score
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_seconds"] = train_seconds
    test_metrics["history"] = history
    test_metrics["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    test_metrics["inference"] = measure_inference(model, graphs, test_windows, edge_features, device)

    # The restored model is also run once more on the validation flows. Its
    # logits there let a prior or threshold correction be tuned on validation
    # and then applied, unchanged, to the test logits (revision analysis).
    val_final = run_epoch(model, graphs, val_windows, edge_features, labels, edge_split, "val",
                          device, windows_per_batch, None, criterion)

    predictions = {
        "edge_ids": test_out["edge_ids"],
        "y_true": test_out["y_true"].astype("int16"),
        "y_pred": test_out["y_pred"].astype("int16"),
        "y_prob": test_out["y_prob"].astype("float16"),
        "y_logit": test_out["y_logit"].astype("float16"),
        "val_edge_ids": val_final["edge_ids"],
        "val_y_true": val_final["y_true"].astype("int16"),
        "val_y_logit": val_final["y_logit"].astype("float16"),
        "class_weights": weights,
    }
    if "gate_mean" in test_out:
        predictions["gate_mean"] = test_out["gate_mean"].astype("float32")
        predictions["gate_nodes"] = test_out["gate_nodes"]
    return model, test_metrics, predictions
