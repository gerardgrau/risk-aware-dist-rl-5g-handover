"""Conceptual figures: the pinball (quantile) loss and its Huber variant.

Two slide-ready, transparent figures in the deck's blue/orange style:

  pinball_loss.png        — the quantile (pinball) loss for low / median / high
                            tau, as a function of the error u = y_true - y_pred.
                            A LOW quantile is steep on the LEFT (penalized hard
                            when the true value lands below it, slope 1-tau) and
                            shallow on the right (slope tau); the high quantile
                            mirrors it; tau=0.5 is the symmetric median loss.
  quantile_huber_loss.png — the same loss with the Huber rounding: the kink at
                            u=0 becomes a quadratic on |u|<=kappa, giving bounded,
                            continuous gradients near zero.

Convention matches the paper: error u = y_true - y_pred = target - prediction,
weight |tau - 1{u<0}| (= tau for u>=0, 1-tau for u<0).

    ./venv-RL/bin/python3 src/tools/plot_pinball_loss.py
    ./venv-RL/bin/python3 src/tools/plot_pinball_loss.py --figs pinball
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# Palette: cool low -> warm high (blue and orange are the deck's accent colours).
C_LOW = "#1f5fd0"    # tau = 0.1
C_MED = "#555555"    # tau = 0.5 (median)
C_HIGH = "#ee8a1a"   # tau = 0.9
AXIS_C = "#cfcfcf"
BAND_C = "#9aa0a6"
TEXT_C = "#111111"

OUT_DIR = "figures/11_qrdqn_explainer"
TAUS = (0.1, 0.5, 0.9)
COLORS = {0.1: C_LOW, 0.5: C_MED, 0.9: C_HIGH}
XMAX = 4.0
KAPPA = 1.0          # Huber threshold (large enough that the knee is visible)

plt.rcParams.update({
    "savefig.dpi": 240,
    "savefig.transparent": True,
    "font.size": 15,
})


def _weight(tau: float, u: np.ndarray) -> np.ndarray:
    """Asymmetric quantile weight |tau - 1{u<0}|: tau for u>=0, 1-tau for u<0."""
    return np.where(u >= 0, tau, 1.0 - tau)


def _pinball(tau: float, u: np.ndarray) -> np.ndarray:
    return _weight(tau, u) * np.abs(u)


def _huber(u: np.ndarray, kappa: float) -> np.ndarray:
    a = np.abs(u)
    return np.where(a <= kappa, 0.5 * u**2, kappa * (a - 0.5 * kappa))


def _qhuber(tau: float, u: np.ndarray, kappa: float) -> np.ndarray:
    return _weight(tau, u) * _huber(u, kappa)


def _label(tau: float) -> str:
    base = rf"$\tau={tau:g}$"
    if tau == 0.5:
        return base + " (median)"
    return base


def _base_ax(ylabel: str):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.axhline(0, color=AXIS_C, lw=1.0, zorder=1)
    ax.axvline(0, color=AXIS_C, lw=1.0, zorder=1)
    ax.set_xlim(-XMAX, XMAX)
    ax.set_xlabel("error $=$ true $-$ predicted", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def fig_pinball(path: str) -> None:
    """Corrected pinball loss: low quantile steep on the left."""
    u = np.linspace(-XMAX, XMAX, 801)
    fig, ax = _base_ax("pinball loss")
    for tau in TAUS:
        ax.plot(u, _pinball(tau, u), color=COLORS[tau], lw=2.8,
                label=_label(tau), zorder=4)
    ax.set_ylim(0, _pinball(0.1, np.array([-XMAX]))[0] * 1.06)
    ax.legend(loc="upper center", frameon=False, fontsize=14, ncol=3,
              columnspacing=1.3, handlelength=1.6)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.13)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_quantile_huber(path: str) -> None:
    """Quantile-Huber loss: the kink at u=0 rounded into a quadratic."""
    u = np.linspace(-XMAX, XMAX, 801)
    fig, ax = _base_ax("quantile-Huber loss")
    # highlight the quadratic region |u| <= kappa
    ax.axvspan(-KAPPA, KAPPA, color=BAND_C, alpha=0.12, zorder=0)
    for tau in TAUS:
        ax.plot(u, _qhuber(tau, u, KAPPA), color=COLORS[tau], lw=2.8,
                label=_label(tau), zorder=4)
    ymax = _qhuber(0.1, np.array([-XMAX]), KAPPA)[0] * 1.10
    ax.set_ylim(0, ymax)
    ax.text(0, ymax * 0.97, r"quadratic  $|u|\leq\kappa$", ha="center",
            va="top", fontsize=12.5, color="#5f6368", zorder=5)
    ax.legend(loc="upper center", frameon=False, fontsize=14, ncol=3,
              columnspacing=1.3, handlelength=1.6,
              bbox_to_anchor=(0.5, 0.90))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.13)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


FIGS = {
    "pinball": ("pinball_loss.png", fig_pinball),
    "huber": ("quantile_huber_loss.png", fig_quantile_huber),
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
