"""Overlay the training-reward curves of the canonical no-gate final agents.

The per-run ``learning_curves.png`` figures show one agent at a time. For a
presentation we want a single axis comparing how each final policy *learns*
over the full 2,000-episode, five-seed schedule. This script plots each agent
as a 50-episode rolling mean with a shaded across-seed std band.

Two figures are written:
- ``finals_learning_overlay.png``     -- the four-agent paper roster
                                         (DQN, RN, Soft-step, RA-1q).
- ``finals_learning_overlay_all.png`` -- the same plus RA (k=7) and hard CVaR.

Run from the repo root with ``PYTHONPATH`` exported (see CLAUDE.md).
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (label, benchmark folder, agent filename prefix, colour).
# Colours are the reserved per-algorithm palette shared with the head-to-head
# figures (DQN orange, RN brown, Soft-step red, Truncated purple, Hard CVaR pink).
_FINALS: list[tuple[str, str, str, str]] = [
    ("Scalar DQN",
     "results/benchmarks/bmk_2026-07-20_7_dqn-6000ep-true", "dqn", "#ff7f0e"),
    ("RN QR-DQN (N=25)",
     "results/benchmarks/bmk_2026-07-21_2_risk-neutral-qr-dqn-n-25-width-512", "qrdqn", "#8c564b"),
    ("Hard CVaR (alpha=0.25)",
     "results/benchmarks/bmk_2026-07-22_2_cvar-risk-dial-a-0.25-width-512", "qrdqn", "#e377c2"),
    ("Truncated QR-DQN (k=7, 2k ep)",
     "results/benchmarks/bmk_2026-07-19_2_truncated-cvar-a-0.25", "qrdqn", "#9467bd"),
    ("Soft-step QR-DQN (Proposed)",
     "results/benchmarks/bmk_2026-07-21_3_soft-step-r-2.0,-width-512", "qrdqn", "#d62728"),
]


def load_train_curves(
    expdir: str, agent: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (episode, mean_reward, std_reward) over seeds for one agent."""
    pattern = os.path.join(expdir, "train", f"{agent}_*_seed*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return np.array([]), np.array([]), np.array([])
    seed_curves = []
    for p in paths:
        df = pd.read_csv(p)
        seed_curves.append(df[["episode", "reward"]].set_index("episode"))
    combined = pd.concat(seed_curves, axis=1)
    mean = combined.mean(axis=1).values
    std = combined.std(axis=1).values
    ep = combined.index.values
    return ep, mean, std


def smooth(y: np.ndarray, w: int = 50) -> np.ndarray:
    """Centered rolling mean to denoise the per-episode reward."""
    if len(y) < w:
        return y
    return pd.Series(y).rolling(w, min_periods=1, center=True).mean().values


def _render(
    variants: list[tuple[str, str, str, str]], title: str, out: str
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, bmk, agent, color in variants:
        ep, mean, std = load_train_curves(bmk, agent)
        if len(ep) == 0:
            print(f"  WARNING missing train curves: {bmk} ({agent})")
            continue
        m = smooth(mean, 50)
        s = smooth(std, 50)
        ax.plot(ep, m, color=color, label=label, linewidth=1.8)
        ax.fill_between(ep, m - s, m + s, color=color, alpha=0.12)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Episode Reward (50-ep rolling mean, ±1σ across 5 seeds)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    os.makedirs("figures/02_learning_curves", exist_ok=True)
    title = "Multi-Seed Training Dynamics (6,000 Episodes, 5 Seeds)"
    out_all = "figures/02_learning_curves/finals_learning_overlay_all.png"
    out_std = "figures/02_learning_curves/finals_learning_overlay.png"
    _render(_FINALS, title, out_all)
    _render(_FINALS, title, out_std)

    # Keep docs/assets synchronized for README.md
    docs_asset = "docs/assets/finals_learning_overlay_all.png"
    if os.path.exists(out_all):
        import shutil
        shutil.copyfile(out_all, docs_asset)
        print(f"Synchronized {docs_asset}")


if __name__ == "__main__":
    main()
