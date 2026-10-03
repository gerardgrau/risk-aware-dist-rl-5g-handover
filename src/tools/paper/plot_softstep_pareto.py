"""Generate publication-quality Pareto Plot for the 3x3 Soft-Step Sensitivity Grid.

X-axis: Handover Failure Rate (HOF/min) - Lower is better
Y-axis: Average Spectral Efficiency / Capacity (bps/Hz) - Higher is better

Visual encoding:
  - Color encodes alpha in {0.10, 0.25, 0.50}
  - Marker shape encodes r in {1.5, 2.0, 3.0}
  - The champion (alpha=0.25, r=2.0) is prominently highlighted.
  - A two-panel version also shows HOF vs Radio Link Failure (RLF/min).

Outputs:
  results/final_metrics/plots/softstep_sensitivity_pareto.png
  results/final_metrics/plots/softstep_sensitivity_pareto_2panel.png
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Color mapping for alpha
ALPHA_COLORS = {
    0.10: "#1f77b4",  # Blue
    0.25: "#e6550d",  # Amber / Red-Orange (Champion family)
    0.50: "#2ca02c",  # Green
}

# Marker mapping for r
R_MARKERS = {
    1.5: "o",  # Circle
    2.0: "s",  # Square
    3.0: "^",  # Triangle
}


def load_grid_data(repo_root: Path) -> pd.DataFrame:
    """Load evaluation summary for all 9 configurations."""
    results_json = repo_root / "results/final_metrics/sweeps/softstep_grid_results.json"
    rows = []
    if results_json.exists():
        with open(results_json) as f:
            entries = json.load(f)
        for entry in entries:
            csv_path = Path(entry["summary_csv"])
            if csv_path.exists():
                df = pd.read_csv(csv_path).set_index("metric")
                rows.append({
                    "alpha": float(entry["alpha"]),
                    "r": float(entry["r"]),
                    "capacity": float(df.loc["capacity_avg", "mean"]),
                    "hof": float(df.loc["hof_rate", "mean"]),
                    "rlf": float(df.loc["rlf_rate", "mean"]),
                    "ho": float(df.loc["ho_rate", "mean"]),
                    "pp": float(df.loc["pp_rate", "mean"]),
                    "reliability": float(df.loc["reliability_pct", "mean"]),
                    "reward": float(df.loc["reward", "mean"]),
                    "n_decisions": float(df.loc["n_decisions", "mean"]),
                    "composite_reward": float(df.loc["composite_reward", "mean"]),
                })
    
    df_all = pd.DataFrame(rows).sort_values(["alpha", "r"]).reset_index(drop=True)
    df_all["rpa"] = df_all["reward"] / df_all["n_decisions"]
    return df_all


def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str, x_minimize: bool = True, y_maximize: bool = True) -> pd.DataFrame:
    """Extract non-dominated Pareto front points."""
    pareto_pts = []
    for i, r1 in df.iterrows():
        dominated = False
        for j, r2 in df.iterrows():
            if i == j:
                continue
            x_better = (r2[x_col] <= r1[x_col]) if x_minimize else (r2[x_col] >= r1[x_col])
            y_better = (r2[y_col] >= r1[y_col]) if y_maximize else (r2[y_col] <= r1[y_col])
            x_strict = (r2[x_col] < r1[x_col]) if x_minimize else (r2[x_col] > r1[x_col])
            y_strict = (r2[y_col] > r1[y_col]) if y_maximize else (r2[y_col] < r1[y_col])
            if (x_better and y_better) and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto_pts.append(r1)
    df_front = pd.DataFrame(pareto_pts)
    if not df_front.empty:
        df_front = df_front.sort_values(x_col)
    return df_front


def plot_single_pareto(df: pd.DataFrame, out_path: Path):
    """Generate 1-panel high-detail Pareto plot (HOF vs Capacity)."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=300)

    # Compute Pareto frontier (minimize HOF, maximize Capacity)
    front = compute_pareto_front(df, "hof", "capacity", x_minimize=True, y_maximize=True)
    if not front.empty:
        ax.plot(
            front["hof"],
            front["capacity"],
            color="#555555",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            zorder=2,
            label="Pareto Frontier (2k ep)",
        )

    # Specific clean label coordinates for each point to prevent any overlap
    label_offsets = {
        (0.10, 1.5): (0, 14),
        (0.10, 2.0): (22, 14),
        (0.10, 3.0): (0, 14),
        (0.25, 1.5): (0, 14),
        (0.25, 2.0): (22, -10),  # Champion
        (0.25, 3.0): (-22, -18),
        (0.50, 1.5): (-28, -18),
        (0.50, 2.0): (28, 12),
        (0.50, 3.0): (-15, -20),
    }

    # Plot all 9 points
    for _, row in df.iterrows():
        a = row["alpha"]
        r = row["r"]
        x = row["hof"]
        y = row["capacity"]
        color = ALPHA_COLORS[a]
        marker = R_MARKERS[r]
        is_champ = (abs(a - 0.25) < 1e-3 and abs(r - 2.0) < 1e-3)
        off_x, off_y = label_offsets.get((round(a, 2), round(r, 1)), (0, 10))

        if is_champ:
            # Halo around champion
            ax.scatter(x, y, s=280, color="none", edgecolors="black", linewidths=2.2, zorder=5)
            ax.scatter(x, y, s=180, color=color, marker=marker, edgecolors="darkred", linewidths=1.5, zorder=6)
            ax.annotate(
                r"$\mathbf{Champion}$" + f"\n" + rf"($\alpha={a:.2f}, r={r:.1f}$)" + f"\n" + r"$\mathbf{Min\ RLF: 0.058}$",
                xy=(x, y),
                xytext=(off_x, off_y),
                textcoords="offset points",
                fontsize=9.5,
                fontweight="bold",
                color="#990000",
                arrowprops=dict(arrowstyle="->", color="#990000", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#e6550d", alpha=0.95),
                zorder=7,
            )
        else:
            ax.scatter(x, y, s=130, color=color, marker=marker, edgecolors="#333333", linewidths=1.0, alpha=0.9, zorder=4)
            ax.annotate(
                rf"$\alpha={a:.2f}, r={r:.1f}$" + f"\n" + rf"(RLF: {row['rlf']:.3f})",
                xy=(x, y),
                xytext=(off_x, off_y),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.85, lw=0.5),
                zorder=4,
            )

    # Set axis limits & labels
    ax.set_xlim(0.68, 1.13)
    ax.set_ylim(4.037, 4.061)
    ax.set_xlabel(r"Handover Failure Rate (HOF / min)  [Lower is Better $\leftarrow$]", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Average Spectral Efficiency (bps / Hz)  [Higher is Better $\uparrow$]", fontsize=11, fontweight="bold")
    ax.set_title("Soft-Step Hyperparameter Sensitivity: Operating Pareto Landscape\n($3 \\times 3$ Grid: $\\alpha \\in \\{0.10, 0.25, 0.50\\} \\times r \\in \\{1.5, 2.0, 3.0\\}$, 2,000 ep, 1,000 UEs)", fontsize=11.5, pad=12)

    # Shaded optimal direction annotation in corner
    ax.text(
        0.03, 0.95,
        "Optimal Region:\nHigh Capacity & Low HOF",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#006600",
        va="top",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#eef9ee", edgecolor="#88cc88", alpha=0.9),
    )

    # Legends: separate for Alpha (colors) and R (shapes)
    legend_elements_alpha = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.10], markersize=9, label=r"$\alpha = 0.10$ (Tail Decile)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.25], markersize=9, label=r"$\alpha = 0.25$ (Lower Quartile)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.50], markersize=9, label=r"$\alpha = 0.50$ (Lower Median)"),
    ]
    legend_elements_r = [
        Line2D([0], [0], marker=R_MARKERS[1.5], color="#444", markerfacecolor="#888", markersize=8, linestyle="None", label=r"$r = 1.5$ (Mild penalty)"),
        Line2D([0], [0], marker=R_MARKERS[2.0], color="#444", markerfacecolor="#888", markersize=8, linestyle="None", label=r"$r = 2.0$ (2:1 balance)"),
        Line2D([0], [0], marker=R_MARKERS[3.0], color="#444", markerfacecolor="#888", markersize=8, linestyle="None", label=r"$r = 3.0$ (Strong penalty)"),
    ]

    leg1 = ax.legend(handles=legend_elements_alpha, loc="lower left", title=r"$\mathbf{Tail\ Cutoff\ (\alpha)}$", fontsize=8.5, title_fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=legend_elements_r, loc="lower right", title=r"$\mathbf{Penalty\ Ratio\ (r)}$", fontsize=8.5, title_fontsize=9, framealpha=0.95)

    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_two_panel_pareto(df: pd.DataFrame, out_path: Path):
    """Generate 2-panel figure: (a) Capacity vs HOF, (b) RLF vs HOF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)

    # Panel (a): Capacity vs HOF
    front1 = compute_pareto_front(df, "hof", "capacity", x_minimize=True, y_maximize=True)
    if not front1.empty:
        ax1.plot(front1["hof"], front1["capacity"], color="#666666", linestyle="--", lw=1.3, alpha=0.7)

    # Panel (b): RLF vs HOF (both minimize!)
    front2 = compute_pareto_front(df, "hof", "rlf", x_minimize=True, y_maximize=False)
    if not front2.empty:
        ax2.plot(front2["hof"], front2["rlf"], color="#666666", linestyle="--", lw=1.3, alpha=0.7)

    offsets_panel_a = {
        (0.10, 1.5): (-24, 10),
        (0.10, 2.0): (22, 10),
        (0.10, 3.0): (0, 10),
        (0.25, 1.5): (0, 10),
        (0.25, 2.0): (20, -14),
        (0.25, 3.0): (-18, -14),
        (0.50, 1.5): (-24, -14),
        (0.50, 2.0): (24, 10),
        (0.50, 3.0): (-12, -16),
    }

    offsets_panel_b = {
        (0.10, 1.5): (-20, 10),
        (0.10, 2.0): (22, 10),
        (0.10, 3.0): (0, 10),
        (0.25, 1.5): (0, 10),
        (0.25, 2.0): (-28, -26),  # Champion placed to the left/down
        (0.25, 3.0): (0, 10),
        (0.50, 1.5): (-22, -14),
        (0.50, 2.0): (22, 8),
        (0.50, 3.0): (-15, -16),
    }

    for _, row in df.iterrows():
        a = row["alpha"]
        r = row["r"]
        x = row["hof"]
        y_cap = row["capacity"]
        y_rlf = row["rlf"]
        color = ALPHA_COLORS[a]
        marker = R_MARKERS[r]
        is_champ = (abs(a - 0.25) < 1e-3 and abs(r - 2.0) < 1e-3)
        off_a = offsets_panel_a.get((round(a, 2), round(r, 1)), (0, 8))
        off_b = offsets_panel_b.get((round(a, 2), round(r, 1)), (0, 8))

        # Draw panel (a)
        if is_champ:
            ax1.scatter(x, y_cap, s=240, color="none", edgecolors="black", linewidths=2.0, zorder=5)
            ax1.scatter(x, y_cap, s=150, color=color, marker=marker, edgecolors="darkred", linewidths=1.5, zorder=6)
            ax1.annotate("Champion\n" + rf"($\alpha={a:.2f}, r={r:.1f}$)", (x, y_cap), xytext=off_a,
                         textcoords="offset points", fontsize=8.5, fontweight="bold", color="#990000",
                         arrowprops=dict(arrowstyle="->", color="#990000", lw=1.0),
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff5f5", edgecolor="#e6550d", alpha=0.9))
        else:
            ax1.scatter(x, y_cap, s=110, color=color, marker=marker, edgecolors="#333", alpha=0.85, zorder=4)
            ax1.annotate(rf"({a:.2f}, {r:.1f})", (x, y_cap), xytext=off_a, textcoords="offset points",
                         fontsize=7.5, ha="center", color="#444")

        # Draw panel (b)
        if is_champ:
            ax2.scatter(x, y_rlf, s=240, color="none", edgecolors="black", linewidths=2.0, zorder=5)
            ax2.scatter(x, y_rlf, s=150, color=color, marker=marker, edgecolors="darkred", linewidths=1.5, zorder=6)
            ax2.annotate(r"$\mathbf{Champion}$" + "\n" + r"$\mathbf{Lowest\ RLF: 0.058}$", (x, y_rlf), xytext=off_b,
                         textcoords="offset points", fontsize=8.5, fontweight="bold", color="#990000",
                         arrowprops=dict(arrowstyle="->", color="#990000", lw=1.1),
                         bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f5", edgecolor="#e6550d", alpha=0.95))
        else:
            ax2.scatter(x, y_rlf, s=110, color=color, marker=marker, edgecolors="#333", alpha=0.85, zorder=4)
            ax2.annotate(rf"({a:.2f}, {r:.1f})", (x, y_rlf), xytext=off_b, textcoords="offset points",
                         fontsize=7.5, ha="center", color="#444")

    # Styling panel (a)
    ax1.set_xlim(0.68, 1.13)
    ax1.set_ylim(4.037, 4.060)
    ax1.set_xlabel(r"Handover Failure Rate (HOF / min)  [$\leftarrow$ Better]", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel(r"Spectral Efficiency (bps / Hz)  [Better $\uparrow$]", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Spectral Efficiency vs. Handover Failures", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Styling panel (b)
    ax2.set_xlim(0.68, 1.13)
    ax2.set_ylim(0.052, 0.100)
    ax2.set_xlabel(r"Handover Failure Rate (HOF / min)  [$\leftarrow$ Better]", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel(r"Radio Link Failure Rate (RLF / min)  [$\leftarrow$ Better]", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Radio Link Drops vs. Handover Failures", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Shaded optimal text in panel (b)
    ax2.text(0.04, 0.12, "Optimal Region:\nMin HOF & Min RLF", transform=ax2.transAxes,
             fontsize=8.5, fontweight="bold", color="#006600",
             bbox=dict(boxstyle="square,pad=0.3", facecolor="#eef9ee", edgecolor="#88cc88", alpha=0.9))

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.10], markersize=8, label=r"$\alpha = 0.10$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.25], markersize=8, label=r"$\alpha = 0.25$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.50], markersize=8, label=r"$\alpha = 0.50$"),
        Line2D([0], [0], marker=R_MARKERS[1.5], color="#444", markerfacecolor="#888", markersize=7, linestyle="None", label=r"$r = 1.5$"),
        Line2D([0], [0], marker=R_MARKERS[2.0], color="#444", markerfacecolor="#888", markersize=7, linestyle="None", label=r"$r = 2.0$"),
        Line2D([0], [0], marker=R_MARKERS[3.0], color="#444", markerfacecolor="#888", markersize=7, linestyle="None", label=r"$r = 3.0$"),
    ]
    ax1.legend(handles=legend_elements, loc="lower left", fontsize=8, ncol=2, framealpha=0.95)

    plt.suptitle(r"Multi-Objective Operating Frontiers for Soft-Step QR-DQN ($3 \times 3$ Sensitivity Grid)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    plots_dir = repo_root / "results/final_metrics/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_grid_data(repo_root)
    if df.empty:
        print("Error: No data loaded.")
        return

    out_single = plots_dir / "softstep_sensitivity_pareto.png"
    out_two = plots_dir / "softstep_sensitivity_pareto_2panel.png"

    plot_single_pareto(df, out_single)
    plot_two_panel_pareto(df, out_two)


if __name__ == "__main__":
    main()
