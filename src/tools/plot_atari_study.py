"""Aggregate and plot the Atari quantile-mode study.

Reads each `results/atari/<game>_<variant>_<ts>/` folder's final_eval.txt
plus the train.csv learning curve, then produces a side-by-side
comparison figure for the chosen game.

Usage:
    ./venv-RL/bin/python3 src/tools/plot_atari_study.py --game Breakout
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS = [
    "qrdqn_midpoint",
    "qrdqn_gauss_legendre",
    "qrdqn_trapezoidal",
    "qrdqn_cvar_full",
    "qrdqn_cvar_truncated",
]

VARIANT_LABELS = {
    "qrdqn_midpoint":       "Midpoint (baseline)",
    "qrdqn_gauss_legendre": "Gauss-Legendre",
    "qrdqn_trapezoidal":    "Trapezoidal",
    "qrdqn_cvar_full":      "CVaR(0.1) full",
    "qrdqn_cvar_truncated": "CVaR(0.1) truncated",
}


def find_run(game: str, variant: str) -> str | None:
    pattern = os.path.join("results", "atari", f"{game}_{variant}_*")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_final(run_dir: str) -> float | None:
    path = os.path.join(run_dir, "final_eval.txt")
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("mean_return="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def read_train_csv(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "train.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def plot_atari(game: str, save_path: str) -> dict[str, float]:
    finals: dict[str, float] = {}
    curves: dict[str, pd.DataFrame] = {}

    for v in VARIANTS:
        run_dir = find_run(game, v)
        if run_dir is None:
            continue
        fe = read_final(run_dir)
        if fe is not None:
            finals[v] = fe
        df = read_train_csv(run_dir)
        if df is not None:
            curves[v] = df

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Atari {game} — quantile-mode study", fontsize=14, fontweight="bold")

    palette = plt.cm.tab10.colors
    x_positions = np.arange(len(VARIANTS))
    values = [finals.get(v, float("nan")) for v in VARIANTS]
    bars = ax_bar.bar(
        x_positions, values,
        color=[palette[i % len(palette)] for i in range(len(VARIANTS))],
    )
    valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if valid:
        best_idx, _ = max(valid, key=lambda iv: iv[1])
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)
    ax_bar.set_title("Final greedy mean return (higher is better)")
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(
        [VARIANT_LABELS[v] for v in VARIANTS], rotation=30, ha="right",
    )
    ax_bar.grid(True, linestyle=":", alpha=0.5, axis="y")

    window = 30
    for i, v in enumerate(VARIANTS):
        df = curves.get(v)
        if df is None or "episode_reward" not in df.columns:
            continue
        smoothed = df["episode_reward"].rolling(window=window, min_periods=1).mean()
        ax_curve.plot(
            df["frame"], smoothed,
            color=palette[i % len(palette)], label=VARIANT_LABELS[v], linewidth=1.5,
        )
    ax_curve.set_title(f"Episode reward (rolling {window}-ep mean) vs training frames")
    ax_curve.set_xlabel("frame")
    ax_curve.set_ylabel("reward")
    ax_curve.legend(fontsize=9, loc="best")
    ax_curve.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {save_path}")
    return finals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Breakout")
    args = parser.parse_args()

    save_path = os.path.join(
        "results", "final_metrics", "plots", f"atari_{args.game}_study.png",
    )
    finals = plot_atari(args.game, save_path)

    print(f"\nFinal greedy returns on Atari {args.game}:")
    for v in VARIANTS:
        fe = finals.get(v)
        if fe is None:
            print(f"  {VARIANT_LABELS[v]:<26}  pending")
        else:
            print(f"  {VARIANT_LABELS[v]:<26}  {fe:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
