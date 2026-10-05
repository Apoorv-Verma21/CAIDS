import json

import numpy as np

from .context import ContextExtractor, context_feature_names


class WindowedGraphs:
    def __init__(self, edge_src, edge_dst, edge_window, edge_offsets,
                 node_offsets, context, window_ids, node_names=None):
        self.edge_src = edge_src
        self.edge_dst = edge_dst
        self.edge_window = edge_window
        self.edge_offsets = edge_offsets
        self.node_offsets = node_offsets
        self.context = context
        self.window_ids = window_ids
        self.node_names = node_names

    @property
    def n_windows(self):
        return len(self.window_ids)

    def window(self, i):
        e0, e1 = self.edge_offsets[i], self.edge_offsets[i + 1]
        n0, n1 = self.node_offsets[i], self.node_offsets[i + 1]
        return {
            "window_id": int(self.window_ids[i]),
            "edge_slice": (int(e0), int(e1)),
            "src": self.edge_src[e0:e1],
            "dst": self.edge_dst[e0:e1],
            "context": self.context[n0:n1],
            "n_nodes": int(n1 - n0),
        }

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "graphs.npz",
            edge_src=self.edge_src,
            edge_dst=self.edge_dst,
            edge_window=self.edge_window,
            edge_offsets=self.edge_offsets,
            node_offsets=self.node_offsets,
            context=self.context,
            window_ids=self.window_ids,
        )

    @classmethod
    def load(cls, directory):
        payload = np.load(directory / "graphs.npz")
        return cls(
            payload["edge_src"],
            payload["edge_dst"],
            payload["edge_window"],
            payload["edge_offsets"],
            payload["node_offsets"],
            payload["context"],
            payload["window_ids"],
        )


def build_windowed_graphs(df, cfg, verbose=True):
    extractor = ContextExtractor(cfg)
    min_flows = int(cfg.data.get("min_flows_per_window", 2))
    max_windows = cfg.data.get("max_windows")

    in_bytes = df["IN_BYTES"].values if "IN_BYTES" in df.columns else np.zeros(len(df))
    out_bytes = df["OUT_BYTES"].values if "OUT_BYTES" in df.columns else np.zeros(len(df))
    total_bytes = np.asarray(in_bytes, dtype="float64") + np.asarray(out_bytes, dtype="float64")

    src_ip = df["src_ip"].values
    dst_ip = df["dst_ip"].values
    src_port = df["src_port"].values
    dst_port = df["dst_port"].values
    start_ms = df["start_ms"].values
    window_col = df["window_id"].values

    boundaries = np.flatnonzero(np.diff(window_col)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(df)]])

    edge_src_all = []
    edge_dst_all = []
    edge_rows_all = []
    context_all = []
    edge_offsets = [0]
    node_offsets = [0]
    window_ids = []

    kept = 0
    for w in range(len(starts)):
        lo, hi = int(starts[w]), int(ends[w])
        if hi - lo < min_flows:
            continue
        if max_windows is not None and kept >= int(max_windows):
            break

        hosts = np.concatenate([src_ip[lo:hi], dst_ip[lo:hi]])
        nodes, inverse = np.unique(hosts, return_inverse=True)
        n_edges = hi - lo
        src_local = inverse[:n_edges].astype("int32")
        dst_local = inverse[n_edges:].astype("int32")

        context = extractor.process_window(
            window_index=int(window_col[lo]),
            nodes=nodes,
            src_local=src_local,
            dst_local=dst_local,
            src_port=src_port[lo:hi],
            dst_port=dst_port[lo:hi],
            total_bytes=total_bytes[lo:hi],
            start_ms=start_ms[lo],
        )

        edge_src_all.append(src_local)
        edge_dst_all.append(dst_local)
        edge_rows_all.append(np.arange(lo, hi, dtype="int64"))
        context_all.append(context)
        edge_offsets.append(edge_offsets[-1] + n_edges)
        node_offsets.append(node_offsets[-1] + len(nodes))
        window_ids.append(int(window_col[lo]))
        kept += 1

        if verbose and kept % 500 == 0:
            print("  processed {} windows".format(kept), flush=True)

    if kept == 0:
        raise ValueError(
            "No windows had at least {} flows. Lower data.min_flows_per_window "
            "or increase data.window_seconds.".format(min_flows)
        )

    graphs = WindowedGraphs(
        edge_src=np.concatenate(edge_src_all),
        edge_dst=np.concatenate(edge_dst_all),
        edge_window=np.repeat(
            np.arange(kept, dtype="int32"), np.diff(edge_offsets)
        ),
        edge_offsets=np.array(edge_offsets, dtype="int64"),
        node_offsets=np.array(node_offsets, dtype="int64"),
        context=np.concatenate(context_all, axis=0),
        window_ids=np.array(window_ids, dtype="int64"),
    )
    row_index = np.concatenate(edge_rows_all)
    return graphs, row_index, context_feature_names(cfg)


def save_meta(directory, meta):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)


def load_meta(directory):
    with open(directory / "meta.json", "r", encoding="utf-8") as handle:
        return json.load(handle)
