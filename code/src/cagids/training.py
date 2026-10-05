import copy

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_metrics


def resolve_device(setting="auto"):
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_windows(graphs, window_indices, edge_features, labels, device):
    src_parts = []
    dst_parts = []
    context_parts = []
    attr_parts = []
    label_parts = []
    node_offset = 0

    for w in window_indices:
        e0, e1 = int(graphs.edge_offsets[w]), int(graphs.edge_offsets[w + 1])
        n0, n1 = int(graphs.node_offsets[w]), int(graphs.node_offsets[w + 1])

        src_parts.append(graphs.edge_src[e0:e1].astype("int64") + node_offset)
        dst_parts.append(graphs.edge_dst[e0:e1].astype("int64") + node_offset)
        context_parts.append(graphs.context[n0:n1])
        attr_parts.append(edge_features[e0:e1])
        label_parts.append(labels[e0:e1])
        node_offset += n1 - n0

    return {
        "src": torch.from_numpy(np.concatenate(src_parts)).to(device),
        "dst": torch.from_numpy(np.concatenate(dst_parts)).to(device),
        "context": torch.from_numpy(np.concatenate(context_parts, axis=0)).to(device),
        "edge_attr": torch.from_numpy(np.concatenate(attr_parts, axis=0)).to(device),
        "labels": torch.from_numpy(np.concatenate(label_parts)).long().to(device),
    }


def class_weights(labels, n_classes, scheme="inverse"):
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    if scheme == "none":
        return np.ones(n_classes, dtype="float32")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return weights.astype("float32")


def run_epoch(model, graphs, window_indices, edge_features, labels, device,
              windows_per_batch, optimizer=None, criterion=None, shuffle=False):
    training = optimizer is not None
    model.train(training)

    order = np.array(window_indices)
    if shuffle:
        order = order[np.random.permutation(len(order))]

    total_loss = 0.0
    total_edges = 0
    all_pred = []
    all_true = []
    all_prob = []

    for start in range(0, len(order), windows_per_batch):
        batch_windows = order[start:start + windows_per_batch]
        batch = collate_windows(graphs, batch_windows, edge_features, labels, device)
        if batch["labels"].numel() == 0:
            continue

        with torch.set_grad_enabled(training):
            logits = model(batch["context"], batch["src"], batch["dst"], batch["edge_attr"])
            loss = criterion(logits, batch["labels"])

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        n = batch["labels"].numel()
        total_loss += float(loss.item()) * n
        total_edges += n

        if not training:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_prob.append(probs)
            all_pred.append(probs.argmax(axis=1))
            all_true.append(batch["labels"].detach().cpu().numpy())

    result = {"loss": total_loss / max(total_edges, 1)}
    if not training and all_true:
        result["y_true"] = np.concatenate(all_true)
        result["y_pred"] = np.concatenate(all_pred)
        result["y_prob"] = np.concatenate(all_prob, axis=0)
    return result


def train_model(model, graphs, splits, edge_features, labels, class_names, cfg, verbose=True):
    device = resolve_device(cfg.train.get("device", "auto"))
    model = model.to(device)

    splits = {k: list(v) for k, v in splits.items()}
    if not splits.get("val"):
        if len(splits["train"]) < 2:
            raise ValueError("Need at least 2 training windows to carve out a validation set.")
        carve = max(1, int(len(splits["train"]) * 0.1))
        splits["val"] = splits["train"][-carve:]
        splits["train"] = splits["train"][:-carve]
        if verbose:
            print("validation split was empty; using the last {} training windows instead".format(carve))
    if not splits.get("test"):
        raise ValueError("Test split is empty. Rebuild graphs with different split settings.")

    weights = class_weights(
        labels[np.concatenate([
            np.arange(graphs.edge_offsets[w], graphs.edge_offsets[w + 1])
            for w in splits["train"]
        ])],
        len(class_names),
        cfg.train.get("class_weighting", "inverse"),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.get("lr", 0.003)),
        weight_decay=float(cfg.train.get("weight_decay", 1e-5)),
    )

    windows_per_batch = int(cfg.train.get("windows_per_batch", 8))
    patience = int(cfg.train.get("patience", 6))
    best_score = -1.0
    best_state = None
    waited = 0

    for epoch in range(1, int(cfg.train.get("epochs", 30)) + 1):
        train_out = run_epoch(
            model, graphs, splits["train"], edge_features, labels, device,
            windows_per_batch, optimizer, criterion, shuffle=True,
        )
        val_out = run_epoch(
            model, graphs, splits["val"], edge_features, labels, device,
            windows_per_batch, None, criterion,
        )

        if "y_true" not in val_out:
            continue
        val_metrics = compute_metrics(
            val_out["y_true"], val_out["y_pred"], class_names, val_out["y_prob"]
        )
        score = val_metrics["macro_f1"]

        if verbose:
            print("epoch {:>3} | train loss {:.4f} | val loss {:.4f} | val macro F1 {:.4f}".format(
                epoch, train_out["loss"], val_out["loss"], score
            ), flush=True)

        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                if verbose:
                    print("early stopping at epoch {}".format(epoch), flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_out = run_epoch(
        model, graphs, splits["test"], edge_features, labels, device,
        windows_per_batch, None, criterion,
    )
    test_metrics = compute_metrics(
        test_out["y_true"], test_out["y_pred"], class_names, test_out["y_prob"]
    )
    test_metrics["best_val_macro_f1"] = best_score
    return model, test_metrics
