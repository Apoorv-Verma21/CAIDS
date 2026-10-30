"""Draws the CAG-IDS architecture figure used in the paper (Fig. 1).

Plain black-and-white block diagram: square boxes, black lines and text, so the
figure prints the same in greyscale."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 7,
})

INK = "black"


def box(ax, x, y, w, h, title, body, body_size=6.3):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.8, edgecolor=INK, facecolor="white"))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=7.6, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.25, body, ha="center", va="top", fontsize=body_size, linespacing=1.3)


def arrow(ax, p0, p1, lw=0.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=INK,
                                 linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad={}".format(rad)))


def line(ax, pts, lw=0.8, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=INK, lw=lw, ls=ls, solid_capstyle="butt")


def label(ax, x, y, text, size=6.2, ha="center", va="center", style="normal"):
    ax.text(x, y, text, fontsize=size, ha=ha, va=va, color=INK, style=style)


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
        "sorted by start time\nIP / port / time used\nfor graph and context", 6.0)
    box(ax, 1.17, 1.08, 1.03, 0.92, "Window graph $G_t$",
        "$\\Delta$ = 60 s; hosts = nodes\nflows = directed edges", 6.0)
    pts = np.array([[1.36, 1.19], [1.55, 1.38], [1.75, 1.22], [1.98, 1.35], [1.68, 1.49]])
    for a, b in [(0, 1), (1, 2), (1, 4), (2, 3), (4, 3), (0, 2)]:
        ax.plot(*zip(pts[a], pts[b]), color=INK, lw=0.5)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, facecolor="white", edgecolor=INK, linewidths=0.6, zorder=3)
    arrow(ax, (1.00, 1.54), (1.17, 1.54))

    # host history ------------------------------------------------------------
    box(ax, 0.05, 2.16, 2.15, 0.72, "Host history $\\mathcal{H}_{t-1}$",
        "per-host EWMA mean and variance ($\\alpha$ = 0.3),\npeers already seen, first window seen", 6.0)

    # context lane --------------------------------------------------------------
    box(ax, 2.40, 1.62, 1.62, 1.26, "Host context $\\mathbf{c}_v$",
        "C1 structural (window $t$): in/out-degree,\npeers, PageRank, clustering\n"
        "C2 behavioural (vs. $\\mathcal{H}_{t-1}$): z-scores,\npeer novelty, age, first-seen\n"
        "C3 role: private address, inbound\nratio, service-port share, port entropy,\ntotal flows", 5.9)
    arrow(ax, (2.20, 2.52), (2.40, 2.52))
    arrow(ax, (2.20, 1.78), (2.40, 1.78))
    # history update: dashed feedback line from the context box back to the history box
    line(ax, [(3.21, 2.88), (3.21, 3.00), (1.12, 3.00)], ls=(0, (3, 2)))
    arrow(ax, (1.12, 3.0), (1.12, 2.88), ls=(0, (3, 2)))
    label(ax, 3.32, 3.00, "history updated with window $t$ after its context is computed", size=5.9, ha="left")

    box(ax, 4.22, 1.92, 1.08, 0.96, "Context encoder",
        "2-layer MLP\n$\\mathbf{c}^{\\prime}_v=\\mathrm{MLP}(\\mathbf{c}_v)$", 6.2)
    arrow(ax, (4.02, 2.40), (4.22, 2.40))

    # topology lane ---------------------------------------------------------------
    box(ax, 4.22, 0.50, 1.08, 1.12, "Edge-aware MP",
        "$K$ = 2 layers\n(mean SAGE or GATv2)\nmessages from $[\\mathbf{h}_u \\| \\mathbf{e}_{uv}]$\nin both directions", 5.9)
    arrow(ax, (4.70, 1.92), (4.70, 1.62))
    label(ax, 4.98, 1.77, "$\\mathbf{h}^{(0)}_v=\\mathbf{c}^{\\prime}_v$")
    line(ax, [(1.69, 1.08), (1.69, 0.82), (4.05, 0.82)])
    arrow(ax, (4.04, 0.82), (4.22, 0.82))
    label(ax, 2.95, 0.73, "edges carry flow features $\\mathbf{e}_{uv}$\n"
          "(IPs, ports, timestamps, TTL and byte rates removed)", size=5.8, va="top")

    # fusion and classifier ------------------------------------------------------
    box(ax, 5.50, 0.50, 1.55, 1.12, "Gated fusion",
        "$\\mathbf{g}_v=\\sigma(W_g[\\mathbf{h}^{(K)}_v \\| \\mathbf{c}^{\\prime}_v]+\\mathbf{b}_g)$\n"
        "$\\mathbf{z}_v=\\mathbf{g}_v\\odot\\mathbf{h}^{(K)}_v$\n$\\;+(1-\\mathbf{g}_v)\\odot(W_c\\mathbf{c}^{\\prime}_v+\\mathbf{b}_c)$", 6.1)
    arrow(ax, (5.30, 1.00), (5.50, 1.00))
    label(ax, 5.40, 1.10, "$\\mathbf{h}^{(K)}_v$")
    line(ax, [(5.30, 2.12), (5.40, 2.12), (5.40, 1.75), (5.75, 1.75)])
    arrow(ax, (5.75, 1.755), (5.75, 1.62))
    label(ax, 5.37, 1.88, "$\\mathbf{c}^{\\prime}_v$", ha="right")

    box(ax, 5.50, 1.92, 1.55, 0.96, "Flow classifier",
        "softmax(MLP$[\\mathbf{z}_u \\| \\mathbf{z}_v \\| \\mathbf{e}_{uv}]$)\nBenign or one of the\n9 attack classes", 6.1)
    arrow(ax, (6.40, 1.62), (6.40, 1.92))
    label(ax, 6.62, 1.77, "$\\mathbf{z}_u,\\mathbf{z}_v$")
    line(ax, [(1.69, 0.82), (1.69, 0.26), (7.15, 0.26), (7.15, 2.30)])
    arrow(ax, (7.155, 2.30), (7.05, 2.30))
    label(ax, 4.40, 0.19, "the flow features $\\mathbf{e}_{uv}$ also enter the classifier directly (residual input)",
          size=5.8, va="top")

    fig.savefig(args.out, dpi=300, metadata={"Software": None})
    print("saved", args.out)


if __name__ == "__main__":
    main()
