"""Bare schematic network diagrams for the QR-DQN presentation.

Every figure shares the SAME trunk (input -> two hidden layers, fully
connected, in the clean blue/black/green node-and-edge style) and differs only
in the output head, so the architectural difference is the only thing that
changes between slides:

  head_dqn.png            — DQN: one (green) output neuron per action.
  head_qrdqn.png          — QR-DQN risk-neutral: one stack per action, many
                            (hollow orange) neurons per stack (N quantiles).
  head_rn_1q.png          — QR-DQN RN-1q: one hollow orange neuron per action.
  head_ra_1q.png          — QR-DQN RA-1q: one solid orange neuron per action.
  head_ra_full.png        — risk-averse, non-truncated: full stack with the
                            used tail neurons solid orange and the rest dimmed.
  head_ra_truncated.png   — risk-averse, truncated: only the solid-orange tail
                            neurons are kept.

DQN output neurons are green (the non-distributional baseline); every QR-DQN
variant is orange, hollow for "all quantiles count the same" and solid for
"used for action selection".

The only labels drawn are the two that make the schematic self-explanatory:
the input layer is annotated R^88 (the 88-dim state) and the output rows are
numbered a_1 .. a_N (one row per action). No titles, captions or other text are
added — those go on the slide by hand. Backgrounds are transparent so they
overlay any slide colour.

    ./venv-RL/bin/python3 src/tools/plot_qrdqn_explainer.py
    ./venv-RL/bin/python3 src/tools/plot_qrdqn_explainer.py --figs dqn qrdqn
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Node palette (matches the reference diagram: blue in, black hidden, green out).
INPUT_C = "#1f5fd0"
HIDDEN_C = "#111111"
GREEN_EDGE = "#1e8c3a"   # DQN output neuron outline (hollow, reference style)
ORANGE_EDGE = "#ee8a1a"  # QR-DQN output neuron outline (hollow)
ORANGE_FILL = "#ee8a1a"  # QR-DQN "used for action selection" neuron (solid)
ORANGE_DARK = "#aa5e0c"  # darker orange outline for the solid neuron
IGN_EDGE = "#bcbcbc"     # learned-but-ignored neuron outline
EDGE_C = "#222222"
TEXT_C = "#111111"       # label colour

OUT_DIR = "figures/11_qrdqn_explainer"

# Schematic layout (data coords in [0, 1]; nodes drawn with scatter so they
# stay circular regardless of the axes aspect, and lines are hidden under the
# white-filled nodes exactly like the reference image).
X_IN, X_H1, X_H2 = 0.07, 0.27, 0.47
N_IN, N_H1, N_H2 = 4, 6, 6
A_ROWS = 3              # actions shown (rows in the QR-DQN head)
N_FULL = 5             # output neurons per action for full QR-DQN (stack depth)
N_TRUNC = 2            # output neurons per action for truncated risk-aware
ROW_GAP = 0.24         # vertical spacing between action rows (front column)
X_HEAD0 = 0.64         # x of the front head column
DDX = 0.024            # per-quantile depth offset, x (the "slight angle")
DDY = 0.016            # per-quantile depth offset, y (shallow -> partial occl.)

# Fixed canvas shared by every figure so the trunk is pixel-identical across
# slides (no tight-crop, which would crop each head differently).
XLIM = (0.00, 0.82)
# Asymmetric vertical margin: lots of empty space on top (so a fullscreen paste
# clears the slide title) and only a thin margin at the bottom.
YLIM = (0.07, 1.12)
FIGSIZE = (9.0, 7.45)

NODE_S = 470           # scatter marker area (points^2)
NODE_LW = 2.4

# Minimal annotations (the only text on the figures).
INPUT_LABEL = r"$\mathbb{R}^{88}$"   # the 88-dim state vector
LABEL_FS = 27          # input-label font size
ACTION_FS = 25         # action-label font size
LABEL_DX = 0.05        # gap to the right of the head stack for action labels

plt.rcParams.update({
    "savefig.dpi": 240,
    "savefig.transparent": True,
})


def _layer_ys(n: int, y_mid: float, gap: float) -> np.ndarray:
    """Vertical centres for ``n`` nodes stacked symmetrically around y_mid."""
    return y_mid + (np.arange(n) - (n - 1) / 2.0) * gap


def _nodes(ax, xs, ys, color: str) -> None:
    ax.scatter(xs, ys, s=NODE_S, facecolors="white", edgecolors=color,
               linewidths=NODE_LW, zorder=3, clip_on=False)


def _connect(ax, x0: float, ys0, pts, lw: float = 0.5,
             alpha: float = 0.55) -> None:
    """Fully connect a source layer at x0 to an arbitrary set of target pts."""
    for y0 in ys0:
        for (x1, y1) in pts:
            ax.plot([x0, x1], [y0, y1], color=EDGE_C, lw=lw, alpha=alpha,
                    zorder=1)


def _new_ax():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    return fig, ax


def _trunk(ax):
    """Draw the shared trunk; return the last hidden layer (x, ys)."""
    ys_in = _layer_ys(N_IN, 0.5, 0.165)
    ys_h1 = _layer_ys(N_H1, 0.5, 0.13)
    ys_h2 = _layer_ys(N_H2, 0.5, 0.13)
    _connect(ax, X_IN, ys_in, [(X_H1, y) for y in ys_h1])
    _connect(ax, X_H1, ys_h1, [(X_H2, y) for y in ys_h2])
    _nodes(ax, np.full(N_IN, X_IN), ys_in, INPUT_C)
    _nodes(ax, np.full(N_H1, X_H1), ys_h1, HIDDEN_C)
    _nodes(ax, np.full(N_H2, X_H2), ys_h2, HIDDEN_C)
    # Annotate the input layer as the 88-dim state vector, just below it.
    ax.text(X_IN, ys_in.min() - 0.075, INPUT_LABEL, fontsize=LABEL_FS,
            ha="center", va="center", color=TEXT_C, zorder=10)
    return X_H2, ys_h2


def _row_ys() -> np.ndarray:
    return _layer_ys(A_ROWS, 0.5, ROW_GAP)


def _neuron(ax, x: float, y: float, z: float, fc: str, ec: str) -> None:
    ax.scatter([x], [y], s=NODE_S, facecolors=fc, edgecolors=ec,
               linewidths=NODE_LW, zorder=z, clip_on=False)


def _label_depth(ax, depth: int, label: str | None) -> None:
    """Mark the per-action stack depth (number of quantile outputs) as ``label``.

    Draws a short dimension bar parallel to the top action row's angled stack
    (up-left of it, clear of the action labels on the right) with ``label`` (e.g.
    "N" for a full head, "k" for a truncated one) centred above it. No-op for a
    single-neuron head, where the count is obvious.
    """
    if depth <= 1 or not label:
        return
    y_top = _row_ys()[2]                              # the a_1 row
    front = np.array([X_HEAD0, y_top])
    back = front + (depth - 1) * np.array([DDX, DDY])
    perp = np.array([-DDY, DDX])
    perp = perp / np.hypot(*perp)                     # unit, points up-left
    a = front + perp * 0.05
    b = back + perp * 0.05
    bar = FancyArrowPatch(tuple(a), tuple(b), arrowstyle="|-|",
                          mutation_scale=5, lw=1.8, color=TEXT_C, zorder=11)
    ax.add_patch(bar)
    mid = (a + b) / 2 + perp * 0.045
    ax.text(mid[0], mid[1], label, fontsize=ACTION_FS, ha="center",
            va="center", color=TEXT_C, zorder=11)


def _head(ax, x_h2, ys_h2, depth: int, used: int | None = None,
          solid_used: bool = False, hollow_c: str = ORANGE_EDGE,
          solid_fc: str = ORANGE_FILL, solid_ec: str = ORANGE_DARK,
          depth_label: str | None = None) -> None:
    """Draw the per-action output head as an angled, partially-occluded stack.

    Each action is a column of ``depth`` neurons receding up-right (the slight
    angle); nearer neurons (small j) are drawn last so they occlude the
    farther ones. Every output neuron is independently fully connected.

    Args:
        depth: neurons drawn per action.
        used: number of FRONT neurons that count for action selection. None
            means "all the same" (plain hollow). When set, the first ``used``
            neurons are highlighted and the rest are dimmed.
        solid_used: fill the highlighted neurons solid (vs hollow).
        hollow_c: outline colour of hollow (white-filled) neurons.
        solid_fc, solid_ec: fill / outline colour of the solid neurons.
    """
    # every output neuron is its own independently fully-connected unit
    targets = [(X_HEAD0 + j * DDX, y + j * DDY)
               for y in _row_ys() for j in range(depth)]
    _connect(ax, x_h2, ys_h2, targets, lw=0.45, alpha=0.45)
    for y in _row_ys():
        for j in range(depth - 1, -1, -1):       # back -> front (occlusion)
            x = X_HEAD0 + j * DDX
            yy = y + j * DDY
            z = 3 + (depth - j)
            is_used = used is None or j < used
            if is_used and solid_used:
                fc, ec = solid_fc, solid_ec
            elif is_used:
                fc, ec = "white", hollow_c
            else:
                fc, ec = "white", IGN_EDGE
            _neuron(ax, x, yy, z, fc, ec)
    _label_actions(ax, depth)
    _label_depth(ax, depth, depth_label)


def _label_actions(ax, depth: int) -> None:
    """Number the per-action rows a_1, a_2, ..., a_N to the right of the head.

    Three rows are drawn as a stand-in for the N actions, so they are labelled
    a_1 / a_2 / a_N with a vertical ellipsis between the last two to signal the
    omitted ones. Labels sit just past the deepest neuron and are nudged up by
    half the stack depth so they stay centred on each (angled) column.
    """
    rows = _row_ys()                       # [bottom, middle, top]
    label_x = X_HEAD0 + (depth - 1) * DDX + LABEL_DX
    y_off = (depth - 1) * DDY / 2.0        # centre on the angled stack
    for y, lab in ((rows[2], r"$a_1$"),
                   (rows[1], r"$a_2$"),
                   (rows[0], r"$a_N$")):
        ax.text(label_x, y + y_off, lab, fontsize=ACTION_FS, ha="left",
                va="center", color=TEXT_C, zorder=10)
    y_mid = (rows[1] + rows[0]) / 2.0 + y_off
    ax.text(label_x, y_mid, r"$\vdots$", fontsize=ACTION_FS, ha="left",
            va="center", color=TEXT_C, zorder=10)


def _render(path: str, depth: int, used: int | None = None,
            solid_used: bool = False, hollow_c: str = ORANGE_EDGE,
            solid_fc: str = ORANGE_FILL, solid_ec: str = ORANGE_DARK,
            depth_label: str | None = None) -> None:
    fig, ax = _new_ax()
    x_h2, ys_h2 = _trunk(ax)
    _head(ax, x_h2, ys_h2, depth, used=used, solid_used=solid_used,
          hollow_c=hollow_c, solid_fc=solid_fc, solid_ec=solid_ec,
          depth_label=depth_label)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_dqn(path: str) -> None:
    """DQN: a single (green) output neuron per action."""
    _render(path, depth=1, hollow_c=GREEN_EDGE)


def fig_qrdqn(path: str) -> None:
    """QR-DQN (risk-neutral): the full return distribution — N neurons/action."""
    _render(path, depth=N_FULL, depth_label="N")


def fig_rn_1q(path: str) -> None:
    """QR-DQN RN-1q: one hollow (orange) neuron per action."""
    _render(path, depth=1)


def fig_ra_1q(path: str) -> None:
    """QR-DQN RA-1q: one solid (orange) neuron per action."""
    _render(path, depth=1, used=1, solid_used=True)


def fig_ra_full(path: str) -> None:
    """Risk-averse, non-truncated: full stack with the used tail highlighted."""
    _render(path, depth=N_FULL, used=N_TRUNC, solid_used=True, depth_label="N")


def fig_ra_truncated(path: str) -> None:
    """Risk-averse, truncated: only the used tail neurons are kept."""
    _render(path, depth=N_TRUNC, used=N_TRUNC, solid_used=True, depth_label="k")


FIGS = {
    "dqn": ("head_dqn.png", fig_dqn),
    "qrdqn": ("head_qrdqn.png", fig_qrdqn),
    "rn_1q": ("head_rn_1q.png", fig_rn_1q),
    "ra_1q": ("head_ra_1q.png", fig_ra_1q),
    "ra_full": ("head_ra_full.png", fig_ra_full),
    "ra_truncated": ("head_ra_truncated.png", fig_ra_truncated),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figs", nargs="+", default=list(FIGS),
                   choices=list(FIGS), help="which head diagrams to render")
    p.add_argument("--out", default=OUT_DIR, help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for key in args.figs:
        fname, fn = FIGS[key]
        fn(os.path.join(args.out, fname))


if __name__ == "__main__":
    main()
