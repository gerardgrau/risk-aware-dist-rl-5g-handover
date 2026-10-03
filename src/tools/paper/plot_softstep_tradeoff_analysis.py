"""Comprehensive 4-panel 2D trade-off analysis of the 3x3 Soft-Step Sensitivity Grid.

Panels:
  (a) RLF vs HOF: Radio Link Drops vs Handover Execution Failures
  (b) Ping-Pong vs HO Rate: Mobility Stability & Control-Plane Signaling
  (c) Resource Reservation (%) vs Total Failures (HOF + RLF)
  (d) Spectral Efficiency (bps/Hz) vs Total Failures (HOF + RLF)

Visual Encoding:
  - Color indicates alpha in {0.10, 0.25, 0.50}
  - Marker indicates r in {1.5, 2.0, 3.0}
  - Pareto frontiers highlighted on all panels.

Outputs:
  results/final_metrics/plots/softstep_tradeoff_4panel.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ALPHA_COLORS = {
    0.10: "#1f77b4",  # Blue
    0.25: "#e6550d",  # Amber / Red-Orange
    0.50: "#2ca02c",  # Green
}

R_MARKERS = {
    1.5: "o",  # Circle
    2.0: "s",  # Square
    3.0: "^",  # Triangle
}


def load_grid_df(repo_root: Path) -> pd.DataFrame:
    results_json = repo_root / "results/final_metrics/sweeps/softstep_grid_results.json"
    rows = []
    with open(results_json) as f:
        entries = json.load(f)
    for entry in entries:
        csv_path = Path(entry["summary_csv"])
        df = pd.read_csv(csv_path).set_index("metric")
        rows.append({
            "alpha": float(entry["alpha"]),
            "r": float(entry["r"]),
            "cap": float(df.loc["capacity_avg", "mean"]),
            "hof": float(df.loc["hof_rate", "mean"]),
            "rlf": float(df.loc["rlf_rate", "mean"]),
            "total_fail": float(df.loc["hof_rate", "mean"] + df.loc["rlf_rate", "mean"]),
            "ho": float(df.loc["ho_rate", "mean"]),
            "pp": float(df.loc["pp_rate", "mean"]),
            "res": float(df.loc["res_reservation_pct", "mean"]),
            "prep": float(df.loc["prep_rate", "mean"]),
        })
    return pd.DataFrame(rows).sort_values(["alpha", "r"]).reset_index(drop=True)


def compute_pareto(df: pd.DataFrame, x_col: str, y_col: str, x_min: bool = True, y_min: bool = True) -> pd.DataFrame:
    pts = []
    for i, r1 in df.iterrows():
        dominated = False
        for j, r2 in df.iterrows():
            if i == j: continue
            xb = (r2[x_col] <= r1[x_col]) if x_min else (r2[x_col] >= r1[x_col])
            yb = (r2[y_col] <= r1[y_col]) if y_min else (r2[y_col] >= r1[y_col])
            xs = (r2[x_col] < r1[x_col]) if x_min else (r2[x_col] > r1[x_col])
            ys = (r2[y_col] < r1[y_col]) if y_min else (r2[y_col] > r1[y_col])
            if (xb and yb) and (xs or ys):
                dominated = True
                break
        if not dominated:
            pts.append(r1)
    df_p = pd.DataFrame(pts)
    if not df_p.empty:
        df_p = df_p.sort_values(x_col)
    return df_p


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    plots_dir = repo_root / "results/final_metrics/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = load_grid_df(repo_root)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # Panel (a): RLF vs HOF (both minimize)
    p_a = compute_pareto(df, "hof", "rlf", True, True)
    ax_a.plot(p_a["hof"], p_a["rlf"], color="#555555", linestyle="--", lw=1.4, zorder=2, label="Pareto Frontier")

    # Panel (b): Ping-Pong vs HO Rate (both minimize)
    p_b = compute_pareto(df, "ho", "pp", True, True)
    ax_b.plot(p_b["ho"], p_b["pp"], color="#555555", linestyle="--", lw=1.4, zorder=2, label="Pareto Frontier")

    # Panel (c): Resource Reservation (%) vs Total Failures (both minimize)
    p_c = compute_pareto(df, "total_fail", "res", True, True)
    ax_c.plot(p_c["total_fail"], p_c["res"], color="#555555", linestyle="--", lw=1.4, zorder=2, label="Pareto Frontier")

    # Panel (d): Spectral Efficiency vs Total Failures (minimize fail, maximize cap)
    p_d = compute_pareto(df, "total_fail", "cap", True, False)
    ax_d.plot(p_d["total_fail"], p_d["cap"], color="#555555", linestyle="--", lw=1.4, zorder=2, label="Pareto Frontier")

    # Plot scatter points on all 4 panels
    for _, r in df.iterrows():
        a = r["alpha"]
        ratio = r["r"]
        color = ALPHA_COLORS[a]
        marker = R_MARKERS[ratio]
        is_champ = (abs(a - 0.25) < 1e-3 and abs(ratio - 2.0) < 1e-3)
        lbl = rf"({a:.2f}, {ratio:.1f})"

        for ax, x_key, y_key in [
            (ax_a, "hof", "rlf"),
            (ax_b, "ho", "pp"),
            (ax_c, "total_fail", "res"),
            (ax_d, "total_fail", "cap"),
        ]:
            x = r[x_key]
            y = r[y_key]
            if is_champ:
                ax.scatter(x, y, s=220, color="none", edgecolors="black", linewidths=2.2, zorder=5)
                ax.scatter(x, y, s=140, color=color, marker=marker, edgecolors="darkred", linewidths=1.5, zorder=6)
            else:
                ax.scatter(x, y, s=110, color=color, marker=marker, edgecolors="#333333", alpha=0.85, zorder=4)

    # Annotate Panel (a): RLF vs HOF
    ax_a.annotate(r"$\mathbf{Champion\ (0.25, 2.0)}$" + "\n" + r"$\mathbf{Min\ RLF: 0.058}$",
                  (0.8806, 0.0580), xytext=(-35, -28), textcoords="offset points",
                  fontsize=8.5, fontweight="bold", color="#990000",
                  arrowprops=dict(arrowstyle="->", color="#990000", lw=1.1),
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f5", edgecolor="#e6550d", alpha=0.95))
    ax_a.annotate(r"$\mathbf{Min\ HOF\ (0.50, 3.0)}$" + "\n" + "HOF: 0.724 (RLF: 0.068)",
                  (0.7240, 0.0684), xytext=(12, 12), textcoords="offset points",
                  fontsize=8.5, color="#006600",
                  arrowprops=dict(arrowstyle="->", color="#006600", lw=1.0),
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#f5fff5", edgecolor="#2ca02c", alpha=0.9))
    ax_a.annotate(r"$(0.10, 3.0)$", (0.7706, 0.0734), xytext=(8, 8), textcoords="offset points", fontsize=8)
    ax_a.annotate(r"$(0.50, 1.5)$", (0.8646, 0.0660), xytext=(-25, -14), textcoords="offset points", fontsize=8)
    ax_a.annotate(r"$(0.50, 2.0)$", (0.9362, 0.0602), xytext=(10, 8), textcoords="offset points", fontsize=8)
    ax_a.annotate(r"$(0.25, 1.5)$ [Worst]", (1.0718, 0.0936), xytext=(-40, 8), textcoords="offset points", fontsize=8, color="darkred")

    ax_a.set_title(r"(a) Connection Reliability: RLF vs. HOF", fontsize=11.5, fontweight="bold")
    ax_a.set_xlabel(r"Handover Failure Rate (HOF / min)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_a.set_ylabel(r"Radio Link Failure Rate (RLF / min)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Annotate Panel (b): Ping-Pong vs HO Rate
    ax_b.annotate(r"$\mathbf{Min\ Signaling\ (0.10, 3.0)}$" + "\n" + "HO: 5.85, PP: 1.21",
                  (5.8518, 1.2150), xytext=(15, 10), textcoords="offset points",
                  fontsize=8.5, color="#1f77b4",
                  arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f7ff", edgecolor="#1f77b4", alpha=0.9))
    ax_b.annotate(r"$(0.50, 3.0)$" + "\nHO: 6.01, PP: 1.45", (6.0138, 1.4468), xytext=(12, -18), textcoords="offset points", fontsize=8)
    ax_b.annotate(r"$(0.25, 2.0)$ [Champ]" + "\nHO: 6.70, PP: 1.70", (6.6956, 1.6978), xytext=(10, -18), textcoords="offset points", fontsize=8, color="#990000")
    ax_b.annotate(r"$(0.50, 1.5)$ [High PP: 2.04]", (6.8136, 2.0416), xytext=(-45, 10), textcoords="offset points", fontsize=8, color="darkred")

    ax_b.set_title(r"(b) Mobility Overhead: Ping-Pong vs. Handover Rate", fontsize=11.5, fontweight="bold")
    ax_b.set_xlabel(r"Handover Rate (HO / min)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_b.set_ylabel(r"Ping-Pong Rate (PP / min)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Annotate Panel (c): Resource Reservation vs Total Failures
    ax_c.annotate(r"$\mathbf{Lowest\ Total\ Failures\ (0.50, 3.0)}$" + "\n" + "Total: 0.792 / min (Res: 9.27%)",
                  (0.7924, 9.2738), xytext=(15, 10), textcoords="offset points",
                  fontsize=8.5, color="#006600",
                  arrowprops=dict(arrowstyle="->", color="#006600", lw=1.0),
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#f5fff5", edgecolor="#2ca02c", alpha=0.9))
    ax_c.annotate(r"$\mathbf{Lowest\ Resource\ Res.\ (0.25, 2.0)}$" + "\n" + "Res: 8.97% (Total: 0.939 / min)",
                  (0.9386, 8.9691), xytext=(12, -22), textcoords="offset points",
                  fontsize=8.5, color="#990000",
                  arrowprops=dict(arrowstyle="->", color="#990000", lw=1.0),
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff5f5", edgecolor="#e6550d", alpha=0.9))
    ax_c.annotate(r"$(0.10, 3.0)$ (0.844 fail)", (0.8440, 9.1699), xytext=(10, 8), textcoords="offset points", fontsize=8)

    ax_c.set_title(r"(c) Network Cost vs. Total Connection Failures (HOF+RLF)", fontsize=11.5, fontweight="bold")
    ax_c.set_xlabel(r"Total Failures (HOF + RLF / min)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_c.set_ylabel(r"Resource Reservation (% Occupancy)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Annotate Panel (d): Spectral Efficiency vs Total Failures
    ax_d.annotate(r"$\mathbf{Technical\ Tie\ in\ Capacity}$" + "\nAll 9 variants within [4.041, 4.056] bps/Hz\n" + r"($\Delta < 0.37\%$, statistically invariant)",
                  (0.95, 4.050), xytext=(-20, 25), textcoords="offset points",
                  fontsize=8.5, color="#333333",
                  bbox=dict(boxstyle="square,pad=0.35", facecolor="#fffde7", edgecolor="#fbc02d", alpha=0.95))
    ax_d.set_title(r"(d) Spectral Efficiency vs. Total Failures (HOF+RLF)", fontsize=11.5, fontweight="bold")
    ax_d.set_xlabel(r"Total Failures (HOF + RLF / min)  [$\leftarrow$ Better]", fontsize=10, fontweight="bold")
    ax_d.set_ylabel(r"Spectral Efficiency (bps / Hz)  [Better $\uparrow$]", fontsize=10, fontweight="bold")
    ax_d.set_ylim(4.037, 4.062)
    ax_d.grid(True, linestyle=":", alpha=0.6)

    # Global legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.10], markersize=8, label=r"$\alpha = 0.10$ (Tail Decile)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.25], markersize=8, label=r"$\alpha = 0.25$ (Lower Quartile)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALPHA_COLORS[0.50], markersize=8, label=r"$\alpha = 0.50$ (Lower Median)"),
        Line2D([0], [0], marker=R_MARKERS[1.5], color="#444", markerfacecolor="#888", markersize=7, linestyle="None", label=r"$r = 1.5$ (Mild)"),
        Line2D([0], [0], marker=R_MARKERS[2.0], color="#444", markerfacecolor="#888", markersize=7, linestyle="None", label=r"$r = 2.0$ (Balanced)"),
        Line2D([0], [0], marker=R_MARKERS[3.0], color="#444", markerfacecolor="#888", markersize=7, linestyle="None", label=r"$r = 3.0$ (Strong)"),
        Line2D([0], [0], color="#555", linestyle="--", lw=1.4, label="Pareto Frontier"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9.5, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(r"Multidimensional Trade-off Landscape for Soft-Step QR-DQN ($3 \times 3$ Grid, 2,000 ep, 1,000 UEs)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_file = plots_dir / "softstep_tradeoff_4panel.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
