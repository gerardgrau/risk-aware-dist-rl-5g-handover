"""Generate publication-ready 2-panel heatmap for 3x3 Soft-Step Sensitivity Grid.

Swept parameters:
  alpha in [0.10, 0.25, 0.50]
  r in [1.5, 2.0, 3.0]

Panels:
  (a) Handover Failure Rate (HOF / min) - Lower is better
  (b) Average Spectral Efficiency (bps / Hz) - Higher is better

Outputs:
  paper_tmlcn/figures/softstep_sensitivity_grid.png
  results/final_metrics/plots/softstep_sensitivity_grid.png
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHAS = [0.10, 0.25, 0.50]
RS = [1.5, 2.0, 3.0]


def load_grid_data(repo_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load HOF rate, capacity, and HO rate into 3x3 arrays."""
    hof_grid = np.full((len(ALPHAS), len(RS)), np.nan)
    cap_grid = np.full((len(ALPHAS), len(RS)), np.nan)
    ho_grid = np.full((len(ALPHAS), len(RS)), np.nan)

    for i, a in enumerate(ALPHAS):
        for j, r in enumerate(RS):
            tag = f"softstep_grid_a{a:.2f}_r{r:.1f}"
            matches = sorted(glob.glob(os.path.join(repo_root, f"results/benchmarks/*{tag}/eval/qrdqn_summary_seed42.csv")), reverse=True)
            if not matches:
                # Also try looking in bmk_2026-07-21_3_soft-step-r-2.0,-width-512 for a=0.25, r=2.0 if grid run isn't done
                matches = sorted(glob.glob(os.path.join(repo_root, f"results/benchmarks/*{tag}*/eval/qrdqn_summary*.csv")), reverse=True)
            if matches:
                df = pd.read_csv(matches[0]).set_index("metric")
                if "hof_rate" in df.index:
                    hof_grid[i, j] = df.loc["hof_rate", "mean"]
                if "capacity_avg" in df.index:
                    cap_grid[i, j] = df.loc["capacity_avg", "mean"]
                if "ho_rate" in df.index:
                    ho_grid[i, j] = df.loc["ho_rate", "mean"]
            else:
                print(f"Warning: no eval summary found for alpha={a}, r={r}")

    return hof_grid, cap_grid, ho_grid


def plot_heatmaps(
    hof_grid: np.ndarray,
    cap_grid: np.ndarray,
    output_paths: list[str],
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    # -------------------------------------------------------------
    # (a) Handover Failure Rate (HOF / min) - Lower is Better
    # -------------------------------------------------------------
    ax1 = axes[0]
    # Blues_r or coolwarm: lower HOF is better (use coolwarm or YlOrRd)
    im1 = ax1.imshow(hof_grid, cmap="YlGnBu_r", aspect="auto")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("HOF Rate (failures / min)", rotation=90)

    ax1.set_xticks(range(len(RS)))
    ax1.set_xticklabels([f"r = {r:.1f}" for r in RS])
    ax1.set_yticks(range(len(ALPHAS)))
    ax1.set_yticklabels([f"α = {a:.2f}" for a in ALPHAS])
    ax1.set_xlabel("Tail Penalty Ratio (r)")
    ax1.set_ylabel("Risk Tail Cutoff Fraction (α)")
    ax1.set_title("(a) Handover Failure Rate (HOF/min) [↓ Better]", pad=10)

    for i in range(len(ALPHAS)):
        for j in range(len(RS)):
            val = hof_grid[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmean(hof_grid) else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold")
                # Highlight champion (alpha=0.25, r=2.0)
                if np.isclose(ALPHAS[i], 0.25) and np.isclose(RS[j], 2.0):
                    rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#d62728", lw=3.0, ls="--")
                    ax1.add_patch(rect)
                    ax1.text(j, i + 0.26, "[Champion]", ha="center", va="center", color="#d62728", fontsize=9, fontweight="bold")

    # -------------------------------------------------------------
    # (b) Spectral Efficiency / Capacity (bps / Hz) - Higher is Better
    # -------------------------------------------------------------
    ax2 = axes[1]
    im2 = ax2.imshow(cap_grid, cmap="viridis", aspect="auto")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Spectral Efficiency (bps / Hz)", rotation=90)

    ax2.set_xticks(range(len(RS)))
    ax2.set_xticklabels([f"r = {r:.1f}" for r in RS])
    ax2.set_yticks(range(len(ALPHAS)))
    ax2.set_yticklabels([f"α = {a:.2f}" for a in ALPHAS])
    ax2.set_xlabel("Tail Penalty Ratio (r)")
    ax2.set_ylabel("Risk Tail Cutoff Fraction (α)")
    ax2.set_title("(b) Average Spectral Efficiency (bps/Hz) [↑ Better]", pad=10)

    for i in range(len(ALPHAS)):
        for j in range(len(RS)):
            val = cap_grid[i, j]
            if not np.isnan(val):
                text_color = "black" if val > np.nanmean(cap_grid) else "white"
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold")
                if np.isclose(ALPHAS[i], 0.25) and np.isclose(RS[j], 2.0):
                    rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#d62728", lw=3.0, ls="--")
                    ax2.add_patch(rect)
                    ax2.text(j, i + 0.26, "[Champion]", ha="center", va="center", color="#d62728", fontsize=9, fontweight="bold")

    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300)
        print(f"Saved figure to {p}")
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    hof_grid, cap_grid, ho_grid = load_grid_data(repo_root)
    print("HOF Grid (failures / min):")
    print(pd.DataFrame(hof_grid, index=[f"a={a}" for a in ALPHAS], columns=[f"r={r}" for r in RS]))
    print("\nCapacity Grid (bps / Hz):")
    print(pd.DataFrame(cap_grid, index=[f"a={a}" for a in ALPHAS], columns=[f"r={r}" for r in RS]))

    out_paths = [
        os.path.join(repo_root, "paper_tmlcn/figures/softstep_sensitivity_grid.png"),
        os.path.join(repo_root, "results/final_metrics/plots/softstep_sensitivity_grid.png"),
    ]
    plot_heatmaps(hof_grid, cap_grid, out_paths)


if __name__ == "__main__":
    main()
