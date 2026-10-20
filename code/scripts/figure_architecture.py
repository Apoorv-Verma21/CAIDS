"""Draws the CAG-IDS architecture figure used in the paper (Fig. 1)."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 7,
})

C_DATA = ("#dbe9f6", "#2c6fad")
C_CTX = ("#dcefdc", "#2e7d32")
C_GNN = ("#fde9d4", "#c25e00")
C_FUSE = ("#ece2f4", "#6a3d9a")
C_OUT = ("#f6d9d9", "#b03030")
C_HIST = ("#fff6cc", "#9a7b00")


def box(ax, x, y, w, h, title, body, colors, body_size=6.3):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                linewidth=1.0, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=7.8, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.25, body, ha="center", va="top", fontsize=body_size, linespacing=1.3)


def arrow(ax, p0, p1, color="#333333", lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad={}".format(rad)))


def line(ax, pts, color="#555555", lw=0.85):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")


def label(ax, x, y, text, color="#333333", size=6.2, ha="center", va="center"):
    ax.text(x, y, text, fontsize=size, ha=ha, va=va, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fig_architecture.png")
    args = parser.parse_args()

    W, H = 7.2, 3.1
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # inputs -----------------------------------------------------------------
    box(ax, 0.05, 1.08, 0.95, 0.92, "NetFlow stream",
        "sorted by start time\nIP / port / time used\nfor graph and context", C_DATA, 6.0)
    box(ax, 1.17, 1.08, 1.03, 0.92, "Window graph $G_t$",
        "$\\Delta$ = 60 s; hosts = nodes\nflows = directed edges", C_DATA, 6.0)
    pts = np.array([[1.36, 1.19], [1.55, 1.38], [1.75, 1.22], [1.98, 1.35], [1.68, 1.49]])
    for a, b in [(0, 1), (1, 2), (1, 4), (2, 3), (4, 3), (0, 2)]:
        ax.plot(*zip(pts[a], pts[b]), color=C_DATA[1], lw=0.6)
    ax.scatter(pts[:, 0], pts[:, 1], s=7, color=C_DATA[1], zorder=3)
    arrow(ax, (1.00, 1.54), (1.17, 1.54))

    # host history ------------------------------------------------------------
    box(ax, 0.05, 2.16, 2.15, 0.72, "Host history $\\mathcal{H}_{t-1}$",
        "per-host EWMA mean and variance ($\\alpha$ = 0.3),\npeers already seen, first window seen", C_HIST, 6.0)

    # context lane --------------------------------------------------------------
    box(ax, 2.40, 1.62, 1.62, 1.26, "Host context $\\mathbf{c}_v$",
        "C1 structural (window $t$): in/out-degree,\npeers, PageRank, clustering\n"
        "C2 behavioural (vs. $\\mathcal{H}_{t-1}$): z-scores,\npeer novelty, age, first-seen\n"
        "C3 role: private address, inbound\nratio, service-port share, port entropy",
        C_CTX, 5.9)
    arrow(ax, (2.20, 2.52), (2.40, 2.52))
    arrow(ax, (2.20, 1.78), (2.40, 1.78))
    arrow(ax, (3.10, 2.88), (1.60, 2.88), color=C_HIST[1], ls="--", rad=0.22)
    label(ax, 3.22, 3.00, "history updated with window $t$ after its context is computed", color=C_HIST[1], size=5.9, ha="left")

    box(ax, 4.22, 1.92, 1.08, 0.96, "Context encoder",
        "2-layer MLP\n$\\mathbf{c}^{\\prime}_v=\\mathrm{MLP}(\\mathbf{c}_v)$", C_CTX, 6.2)
    arrow(ax, (4.02, 2.40), (4.22, 2.40))

    # topology lane ---------------------------------------------------------------
    box(ax, 4.22, 0.50, 1.08, 1.12, "Edge-aware MP",
        "$K$ = 2 layers\n(mean SAGE or GATv2)\nmessages from $[\\mathbf{h}_u \\| \\mathbf{e}_{uv}]$\nin both directions",
        C_GNN, 5.9)
    arrow(ax, (4.70, 1.92), (4.70, 1.62))
    label(ax, 4.98, 1.77, "$\\mathbf{h}^{(0)}_v=\\mathbf{c}^{\\prime}_v$")
    line(ax, [(1.69, 1.08), (1.69, 0.82), (4.05, 0.82)])
    arrow(ax, (4.04, 0.82), (4.22, 0.82))
    label(ax, 2.95, 0.73, "edges carry flow features $\\mathbf{e}_{uv}$ (IPs, ports, timestamps, TTL removed)",
          color="#444444", size=5.8, va="top")

    # fusion and classifier ------------------------------------------------------
    box(ax, 5.50, 0.50, 1.55, 1.12, "Gated fusion",
        "$\\mathbf{g}_v=\\sigma(W_g[\\mathbf{h}^{(K)}_v \\| \\mathbf{c}^{\\prime}_v]+\\mathbf{b}_g)$\n"
        "$\\mathbf{z}_v=\\mathbf{g}_v\\odot\\mathbf{h}^{(K)}_v$\n$\\;+(1-\\mathbf{g}_v)\\odot(W_c\\mathbf{c}^{\\prime}_v+\\mathbf{b}_c)$",
        C_FUSE, 6.1)
    arrow(ax, (5.30, 1.00), (5.50, 1.00))
    label(ax, 5.40, 1.10, "$\\mathbf{h}^{(K)}_v$")
    line(ax, [(5.30, 2.12), (5.40, 2.12), (5.40, 1.74)], color="#333333", lw=0.9)
    arrow(ax, (5.40, 1.75), (5.60, 1.62))
    label(ax, 5.37, 1.80, "$\\mathbf{c}^{\\prime}_v$", ha="right")

    box(ax, 5.50, 1.92, 1.55, 0.96, "Flow classifier",
        "softmax(MLP$[\\mathbf{z}_u \\| \\mathbf{z}_v \\| \\mathbf{e}_{uv}]$)\nBenign or one of the\n9 attack classes", C_OUT, 6.1)
    arrow(ax, (6.27, 1.62), (6.27, 1.92))
    label(ax, 6.50, 1.77, "$\\mathbf{z}_u,\\mathbf{z}_v$")
    line(ax, [(1.69, 0.82), (1.69, 0.26), (7.15, 0.26), (7.15, 2.30)])
    arrow(ax, (7.155, 2.30), (7.05, 2.30))
    label(ax, 4.40, 0.19, "the flow features $\\mathbf{e}_{uv}$ also enter the classifier directly (residual input)",
          color="#444444", size=5.8, va="top")

    fig.savefig(args.out, dpi=300)
    print("saved", args.out)


if __name__ == "__main__":
    main()
