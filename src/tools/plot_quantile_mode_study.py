"""Aggregate and plot the LTM quantile-mode study.

Auto-discovers the most recent `bmk_*_qmode_<variant>/` folders under
``results/benchmarks/``, reads each variant's eval-summary CSV, and:

    1. Prints a comparison table to stdout.
    2. Saves a 3x3 grid of bar charts (one per metric) to
       ``results/final_metrics/plots/quantile_mode_study.png``.
    3. Optionally appends a row per variant to
       ``results/final_metrics/sweeps/quantile_mode_study.csv`` for later
       master-comparison plotting.

Usage:
    ./venv-RL/bin/python3 src/tools/plot_quantile_mode_study.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS = [
    "qmode_midpoint",
    "qmode_gauss_legendre",
    "qmode_simpson",
    "qmode_trapezoidal",
    "qmode_cvar_full",
    "qmode_cvar_truncated",
]

VARIANT_LABELS = {
    "qmode_midpoint":       "Midpoint (baseline)",
    "qmode_gauss_legendre": "Gauss-Legendre",
    "qmode_simpson":        "Simpson 1/3",
    "qmode_trapezoidal":    "Trapezoidal",
    "qmode_cvar_full":      "CVaR(0.1) full",
    "qmode_cvar_truncated": "CVaR(0.1) truncated",
}

METRICS = [
    ("capacity_avg",          "Capacity (bps/Hz)", "high"),
    ("hof_rate",              "HOF rate (/min)",    "low"),
    ("rlf_rate",              "RLF rate (/min)",    "low"),
    ("ho_rate",               "HO rate (/min)",     "low"),
    ("pp_rate",               "PP rate (/min)",     "low"),
    ("reliability_pct",       "Reliability (%)",    "high"),
    ("prep_rate",             "Prep rate (/min)",   "low"),
    ("res_reservation_pct",   "Res. reservation %", "low"),
    ("reward",                "Episode reward",     "high"),
]


BASELINE_CSV = os.path.join(
    "results", "final_metrics", "baseline_summary.csv",
)


def find_run(variant: str) -> str | None:
    pattern = os.path.join("results", "benchmarks", f"bmk_*_{variant}")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_baseline() -> dict[str, float]:
    """Read the LTM hardcoded-heuristic baseline summary (if present)."""
    if not os.path.exists(BASELINE_CSV):
        return {}
    df = pd.read_csv(BASELINE_CSV)
    df.columns = [c.strip() for c in df.columns]
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row.get("metric", "")).strip()
        try:
            out[name] = float(row["mean"])
        except (ValueError, KeyError):
            continue
    return out


def read_summary(run_dir: str, seed: int = 42) -> dict[str, float]:
    csv_path = os.path.join(
        run_dir, "eval", f"qrdqn_summary_seed{seed}.csv"
    )
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row.get("metric", "")).strip()
        try:
            out[name] = float(row["mean"])
        except (ValueError, KeyError):
            continue
    return out


def print_table(results: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'variant':<22}"
        + "".join(f"{m[0]:>14}" for m in METRICS)
    )
    print(header)
    print("-" * len(header))
    for v in VARIANTS:
        metrics = results.get(v, {})
        cells = "".join(
            f"{metrics.get(m[0], float('nan')):>14.4f}" for m in METRICS
        )
        print(f"{v:<22}{cells}")


def plot_grid(
    results: dict[str, dict[str, float]],
    baseline: dict[str, float],
    save_path: str,
) -> None:
    cols = 3
    rows = (len(METRICS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.2))
    fig.suptitle(
        "Quantile-mode study: per-metric comparison (1 seed × 2000 ep)",
        fontsize=14, fontweight="bold",
    )

    palette = plt.cm.tab10.colors
    x_positions = np.arange(len(VARIANTS))

    for idx, (metric, label, direction) in enumerate(METRICS):
        ax = axes[idx // cols][idx % cols]
        values = [results.get(v, {}).get(metric, float("nan")) for v in VARIANTS]
        bars = ax.bar(
            x_positions, values,
            color=[palette[i % len(palette)] for i in range(len(VARIANTS))],
        )

        # Annotate the best variant for this metric.
        valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        if valid:
            if direction == "high":
                best_idx, _ = max(valid, key=lambda iv: iv[1])
            else:
                best_idx, _ = min(valid, key=lambda iv: iv[1])
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)

        # Overlay the LTM hardcoded-heuristic baseline as a reference line.
        baseline_val = baseline.get(metric)
        if baseline_val is not None:
            ax.axhline(
                baseline_val, color="#444444", linestyle="--", linewidth=1.4,
                label=f"LTM baseline = {baseline_val:.3f}",
            )
            ax.legend(fontsize=7, loc="best", framealpha=0.85)

        ax.set_title(f"{label}  ({'↑ higher better' if direction == 'high' else '↓ lower better'})")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [VARIANT_LABELS[v] for v in VARIANTS],
            rotation=30, ha="right", fontsize=8,
        )
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Hide any unused panels (e.g. 9 metrics in 3x3 grid leaves 0 unused).
    for idx in range(len(METRICS), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {save_path}")


def write_csv(results: dict[str, dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["variant"] + [m[0] for m in METRICS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in VARIANTS:
            row: dict[str, Any] = {"variant": v}
            row.update(results.get(v, {}))
            writer.writerow(row)
    print(f"CSV saved to {path}")


def main() -> int:
    results: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for v in VARIANTS:
        run_dir = find_run(v)
        if run_dir is None:
            missing.append(v)
            continue
        summary = read_summary(run_dir)
        if not summary:
            missing.append(f"{v} (no eval CSV in {run_dir})")
            continue
        results[v] = summary

    if missing:
        print("Missing or incomplete:")
        for m in missing:
            print(f"  - {m}")
        if not results:
            return 1

    baseline = read_baseline()
    if not baseline:
        print(
            f"Note: {BASELINE_CSV} not found — plot will not include the "
            "LTM baseline reference line."
        )

    print()
    print_table(results)

    out_dir = os.path.join("results", "final_metrics")
    plot_grid(results, baseline,
              os.path.join(out_dir, "plots", "quantile_mode_study.png"))
    write_csv(results, os.path.join(out_dir, "sweeps", "quantile_mode_study.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
