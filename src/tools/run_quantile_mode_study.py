"""Sweep QR-DQN quantile-positioning modes on the LTM env.

Runs the six configs under ``configs/quantile_study/`` sequentially:

    1. qmode_midpoint        (baseline)
    2. qmode_gauss_legendre  (non-uniform tau + GL weights)
    3. qmode_simpson         (uniform tau + Simpson 1/3 weights + endpoints)
    4. qmode_trapezoidal     (uniform-with-endpoints + fixed q_min/q_max)
    5. qmode_cvar_full       (midpoint + CVaR(0.1) action selection)
    6. qmode_cvar_truncated  (only learn the bottom-k quantiles)

After all variants finish, prints a comparison table built from each run's
final-eval ``qrdqn_summary_seed42.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time

CONFIGS = [
    "qmode_midpoint",
    "qmode_gauss_legendre",
    "qmode_simpson",
    "qmode_trapezoidal",
    "qmode_cvar_full",
    "qmode_cvar_truncated",
]

CONFIG_DIR = "configs/quantile_study"
RESULTS_DIR = "results/benchmarks"
KEY_METRICS = (
    "capacity_avg",
    "hof_rate",
    "rlf_rate",
    "ho_rate",
    "pp_rate",
    "reliability_pct",
)


def find_latest_run(description: str) -> str | None:
    pattern = os.path.join(RESULTS_DIR, f"bmk_*_{description}")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_summary(run_dir: str, seed: int = 42) -> dict[str, float]:
    csv_path = os.path.join(
        run_dir, "eval", f"qrdqn_summary_seed{seed}.csv"
    )
    if not os.path.exists(csv_path):
        return {}
    out: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("metric") or row.get("Metric")
            if name is None:
                continue
            try:
                out[name] = float(row.get("mean", "nan"))
            except ValueError:
                continue
    return out


def run_one(variant: str, device: str) -> tuple[float, str | None]:
    cfg_path = os.path.join(CONFIG_DIR, f"{variant}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)

    cmd = [
        "venv-RL/bin/python3",
        "src/main.py",
        "--config", cfg_path,
        "--agents", "qrdqn",
        "--device", device,
        "--description", variant,
    ]
    print(f"\n>>> Running variant {variant!r}")
    print("    " + " ".join(cmd))

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    return elapsed, find_latest_run(variant)


def print_comparison(rows: list[tuple[str, float, dict[str, float]]]) -> None:
    print("\n" + "=" * 100)
    print("QUANTILE-MODE STUDY SUMMARY")
    print("=" * 100)
    header = (
        f"{'variant':<26}{'time(s)':>10}  "
        + "".join(f"{m:>14}" for m in KEY_METRICS)
    )
    print(header)
    print("-" * len(header))
    for name, dur, metrics in rows:
        cells = "".join(
            f"{metrics.get(m, float('nan')):>14.4f}" for m in KEY_METRICS
        )
        print(f"{name:<26}{dur:>10.1f}  {cells}")
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="xpu", choices=["cpu", "cuda", "xpu"],
        help="Compute device for each run.",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated subset of variant names to run.",
    )
    args = parser.parse_args()

    variants = CONFIGS if args.only is None else [
        v.strip() for v in args.only.split(",") if v.strip()
    ]

    print(f"=== Quantile-mode study ===")
    print(f"Variants: {variants}")
    print(f"Device:   {args.device}")
    print(f"Budget:   1 seed x num_episodes from each YAML")

    rows: list[tuple[str, float, dict[str, float]]] = []
    for variant in variants:
        try:
            dur, run_dir = run_one(variant, args.device)
        except subprocess.CalledProcessError as err:
            print(f"!!! variant {variant!r} failed: {err}", file=sys.stderr)
            rows.append((variant, float("nan"), {}))
            continue

        metrics = read_summary(run_dir) if run_dir else {}
        rows.append((variant, dur, metrics))
        print(f"--- {variant!r} done in {dur:.1f}s; results -> {run_dir}")

    print_comparison(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
