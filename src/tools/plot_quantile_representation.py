"""Conceptual figures: representing a return distribution by N quantiles.

Two slide-ready, transparent figures that motivate the QR-DQN head, in the same
orange/dark style as ``plot_qrdqn_explainer.py``:

  quantile_pdf_slices.png   — (a) the return PDF carved into N equal-probability
                              slices, with the per-slice quantile values
                              theta_i marked. Shows "equal probability, unequal
                              spacing": tail slices are wide, central ones narrow.
  quantile_staircase.png    — (b) the quantile function F^{-1}(tau): the smooth
                              true inverse-CDF overlaid with the N-quantile
                              staircase approximation, theta_i = F^{-1}(tau_i).

Both use a standard-normal return purely as an illustrative shape (symmetric, so
the equal-area-but-unequal-width point is visible without distraction).

    ./venv-RL/bin/python3 src/tools/plot_quantile_representation.py
    ./venv-RL/bin/python3 src/tools/plot_quantile_representation.py --figs pdf
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Palette shared with the head-diagram figures.
CURVE_C = "#222222"      # the true distribution / quantile function
ORANGE_EDGE = "#ee8a1a"  # quantile markers / approximation
ORANGE_FILL = "#f6b14e"  # slice shading
TEXT_C = "#111111"
GRID_C = "#cfcfcf"
MEAN_C = "#555555"       # risk-neutral mean line
CVAR_C = "#d9730a"       # risk-averse CVaR line

OUT_DIR = "figures/11_qrdqn_explainer"
N = 5                    # quantiles shown (matches the N=5 head stack)
XLO, XHI = -3.2, 3.2     # finite return range for the (otherwise infinite) tails
ALPHA = 0.25             # CVaR tail fraction shown (the paper's knee)

plt.rcParams.update({
    "savefig.dpi": 240,
    "savefig.transparent": True,
    "font.size": 15,
})


def _quantiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tau_mid, theta, edges_x) for the N equal-probability slices."""
    tau_edges = np.arange(N + 1) / N                  # 0, 1/N, ..., 1
    tau_mid = (np.arange(N) + 0.5) / N                # midpoints (i-0.5)/N
    edges_x = norm.ppf(np.clip(tau_edges, 1e-3, 1 - 1e-3))
    edges_x[0], edges_x[-1] = XLO, XHI               # clip the infinite ends
    theta = norm.ppf(tau_mid)                         # theta_i = F^{-1}(tau_i)
    return tau_mid, theta, edges_x


def _theta_label(i: int) -> str:
    """a_1, a_2, ..., a_N style labelling: first two, ellipsis, last."""
    if i == 0:
        return r"$\theta_1$"
    if i == 1:
        return r"$\theta_2$"
    if i == N - 1:
        return r"$\theta_N$"
    return ""


def fig_pdf_slices(path: str) -> None:
    """(a) PDF carved into N equal-probability slices with theta_i marked."""
    _, theta, edges_x = _quantiles()
    xs = np.linspace(XLO, XHI, 600)
    pdf = norm.pdf(xs)

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.plot(xs, pdf, color=CURVE_C, lw=3.0, zorder=5)

    for i in range(N):
        lo, hi = edges_x[i], edges_x[i + 1]
        m = (xs >= lo) & (xs <= hi)
        xseg = np.concatenate(([lo], xs[m], [hi]))
        yseg = np.concatenate(([norm.pdf(lo)], pdf[m], [norm.pdf(hi)]))
        alpha = 0.42 if i % 2 == 0 else 0.22
        ax.fill_between(xseg, 0, yseg, color=ORANGE_FILL, alpha=alpha,
                        lw=0, zorder=2)
        # each slice carries 1/N of the probability mass
        ax.text((lo + hi) / 2.0, 0.014, r"$\frac{1}{N}$", ha="center",
                va="bottom", fontsize=16, color="#7a4a08", zorder=6)
        # quantile value of this slice: stem to the curve + dot on the axis
        ax.plot([theta[i], theta[i]], [0, norm.pdf(theta[i])],
                color=ORANGE_EDGE, lw=1.8, zorder=4)
        ax.scatter([theta[i]], [0], s=90, facecolors=ORANGE_EDGE,
                   edgecolors=ORANGE_EDGE, zorder=7, clip_on=False)
        lab = _theta_label(i)
        if lab:
            ax.text(theta[i], -0.05, lab, ha="center", va="top",
                    fontsize=21, color=TEXT_C, zorder=7)
    # ellipsis between theta_2 and theta_N
    ax.text((theta[1] + theta[-1]) / 2.0, -0.05, r"$\cdots$", ha="center",
            va="top", fontsize=21, color=TEXT_C, zorder=7)

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, norm.pdf(0) * 1.12)
    # axis name at the right end (centred would collide with the theta row)
    ax.text(XHI, -0.055, "return", ha="right", va="top", fontsize=19,
            color=TEXT_C)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, length=0)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.24)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_quantile_staircase(path: str) -> None:
    """(b) inverse-CDF F^{-1}(tau) with the N-quantile staircase overlaid."""
    tau_mid, theta, _ = _quantiles()
    tau_edges = np.arange(N + 1) / N
    tt = np.linspace(0.012, 0.988, 500)
    yy = norm.ppf(tt)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(tt, yy, color=CURVE_C, lw=3.0, zorder=5,
            label=r"true  $F^{-1}(\tau)$")

    # staircase: flat at theta_i across each [edge_i, edge_{i+1}] bin
    for i in range(N):
        lo, hi = tau_edges[i], tau_edges[i + 1]
        ax.plot([lo, hi], [theta[i], theta[i]], color=ORANGE_EDGE, lw=3.0,
                solid_capstyle="butt", zorder=4,
                label=(r"$N$-quantile approx." if i == 0 else None))
        if i < N - 1:                                # vertical riser
            ax.plot([hi, hi], [theta[i], theta[i + 1]], color=ORANGE_EDGE,
                    lw=1.4, alpha=0.6, zorder=3)
    ax.scatter(tau_mid, theta, s=80, facecolors=ORANGE_EDGE,
               edgecolors=ORANGE_EDGE, zorder=7)

    # tau_i ticks at the midpoints, labelled tau_1, tau_2, ..., tau_N
    xt_lab = [r"$\tau_1$", r"$\tau_2$"] + [""] * (N - 3) + [r"$\tau_N$"]
    ax.set_xticks(tau_mid)
    ax.set_xticklabels(xt_lab, fontsize=15)
    ax.text((tau_mid[1] + tau_mid[-1]) / 2.0, ax.get_ylim()[0], "",
            ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(norm.ppf(0.012) * 1.08, norm.ppf(0.988) * 1.08)
    ax.set_xlabel(r"quantile fraction $\tau$", fontsize=16)
    ax.set_ylabel("return", fontsize=16)
    ax.axhline(0, color=GRID_C, lw=1.0, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.13)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_risk_motivation(path: str) -> None:
    """Why CVaR: two actions with ~equal mean but very different worst cases.

    A (aggressive) usually pays well but carries a small lump of rare
    catastrophic failure; B (conservative) gives slightly less on average with no
    bad tail. A has the *higher mean*, so a mean-maximizer prefers it -- yet B is
    obviously safer. The mean cannot tell them apart; the lower tail can.
    """
    RISK_A = "#e8743b"   # aggressive (warm)
    RISK_B = "#2b7bba"   # conservative (cool)
    xs = np.linspace(-4.0, 2.6, 700)
    pdf_a = 0.82 * norm.pdf(xs, 0.8, 0.40) + 0.18 * norm.pdf(xs, -2.3, 0.45)
    pdf_b = norm.pdf(xs, 0.15, 0.55)
    mean_a = 0.82 * 0.8 + 0.18 * (-2.3)      # = 0.242
    mean_b = 0.15
    ytop = max(pdf_a.max(), pdf_b.max())

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.fill_between(xs, 0, pdf_a, color=RISK_A, alpha=0.18, lw=0, zorder=2)
    ax.fill_between(xs, 0, pdf_b, color=RISK_B, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, pdf_a, color=RISK_A, lw=3.0, zorder=5, label="A (aggressive)")
    ax.plot(xs, pdf_b, color=RISK_B, lw=3.0, zorder=5, label="B (conservative)")

    # mean lines (A's is slightly higher); labels offset to opposite sides
    ax.plot([mean_a, mean_a], [0, ytop], color=RISK_A, lw=2.0, ls="--", zorder=4)
    ax.plot([mean_b, mean_b], [0, ytop], color=RISK_B, lw=2.0, ls="--", zorder=4)
    ax.text(mean_b - 0.15, ytop * 1.12, "mean B", color=RISK_B, fontsize=13,
            ha="right", va="center", fontweight="bold")
    ax.text(mean_a + 0.15, ytop * 1.12, "mean A", color=RISK_A, fontsize=13,
            ha="left", va="center", fontweight="bold")
    ax.text((mean_a + mean_b) / 2, ytop * 1.04, "nearly equal",
            color="#777777", fontsize=10, ha="center", va="center")

    # the rare-failure lump of A
    ax.annotate("rare\ncatastrophic\nfailure", xy=(-2.3, 0.17),
                xytext=(-3.0, 0.52), fontsize=12.5, color=RISK_A, ha="center",
                va="center", zorder=7,
                arrowprops=dict(arrowstyle="->", color=RISK_A, lw=1.5))

    # curve labels: A on its right shoulder, B on its left shoulder (clear of
    # the centre cluster).
    ax.text(1.35, pdf_a.max() * 0.45, "A", color=RISK_A, fontsize=20,
            fontweight="bold", ha="center")
    ax.text(-0.7, norm.pdf(-0.7, 0.15, 0.55) + 0.02, "B", color=RISK_B,
            fontsize=20, fontweight="bold", ha="center")

    ax.set_xlim(-4.0, 2.6)
    ax.set_ylim(0, ytop * 1.18)
    ax.text(2.6, -0.018, "return", ha="right", va="top", fontsize=15,
            color=TEXT_C)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, length=0)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.07)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_cvar(path: str) -> None:
    """CVaR_alpha vs the mean: the mean of the worst alpha-fraction of returns."""
    xs = np.linspace(XLO, XHI, 600)
    pdf = norm.pdf(xs)
    q = norm.ppf(ALPHA)                # alpha-quantile = the tail boundary
    cvar = -norm.pdf(q) / ALPHA        # mean of the lower alpha-tail (normal)
    ymax = norm.pdf(0.0)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(xs, pdf, color=CURVE_C, lw=3.0, zorder=5)

    # shade the worst alpha-fraction (the lower tail, return <= q)
    m = xs <= q
    ax.fill_between(np.concatenate(([XLO], xs[m], [q])), 0,
                    np.concatenate(([norm.pdf(XLO)], pdf[m], [norm.pdf(q)])),
                    color=ORANGE_FILL, alpha=0.6, lw=0, zorder=2)
    ax.text(-1.0, 0.12, "worst\n" + rf"$\alpha={ALPHA:g}$",
            ha="center", va="bottom", fontsize=13, color="#7a4a08", zorder=6)

    # risk-neutral: the mean over the WHOLE distribution
    ax.plot([0, 0], [0, ymax], color=MEAN_C, lw=2.2, ls="--", zorder=4)
    ax.text(0, ymax * 1.04, r"mean  $\mathbb{E}[Z]$", ha="center", va="bottom",
            fontsize=15, color=MEAN_C, zorder=6)

    # risk-averse: CVaR = the mean of the shaded tail (sits inside it)
    ax.plot([cvar, cvar], [0, norm.pdf(cvar)], color=CVAR_C, lw=3.0, zorder=6)
    ax.scatter([cvar], [0], s=80, color=CVAR_C, zorder=7, clip_on=False)
    ax.text(cvar, norm.pdf(cvar) + 0.03, r"$\mathrm{CVaR}_\alpha$",
            ha="center", va="bottom", fontsize=17, color=CVAR_C,
            fontweight="bold", zorder=7)
    ax.annotate("mean of the\nshaded tail", xy=(cvar, 0.012),
                xytext=(cvar - 1.15, 0.16), fontsize=12, color=CVAR_C,
                ha="center", va="center", zorder=7,
                arrowprops=dict(arrowstyle="->", color=CVAR_C, lw=1.4))

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, ymax * 1.18)
    ax.text(XHI, -0.012, "return", ha="right", va="top", fontsize=15,
            color=TEXT_C)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, length=0)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.91, bottom=0.07)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


FIGS = {
    "pdf": ("quantile_pdf_slices.png", fig_pdf_slices),
    "staircase": ("quantile_staircase.png", fig_quantile_staircase),
    "cvar": ("cvar_definition.png", fig_cvar),
    "motivation": ("risk_motivation.png", fig_risk_motivation),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figs", nargs="+", default=list(FIGS), choices=list(FIGS))
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key in args.figs:
        fname, fn = FIGS[key]
        fn(os.path.join(args.out, fname))


if __name__ == "__main__":
    main()
