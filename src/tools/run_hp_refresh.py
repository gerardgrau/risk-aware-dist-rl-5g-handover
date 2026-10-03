"""Focused HP refresh after the target-net + QR-quadrature fixes.

Re-tests the HP champion (lr=1e-4, tau=0.01) and a small set of
target-net-sensitive variants under the corrected QR-DQN code:

    baseline_refresh   lr=1e-4, tau=0.01 (champion)
    lr_1e-3            lr=1e-3 (previous default)
    lr_3e-4            lr=3e-4 (previous Phase-1 tie)
    tau_0005           tau=0.005 (slower target lag)
    tau_005            tau=0.05  (faster target lag)
    hard_update        tau=1.0, target_update_freq=1000

Budget: 500 ep x 1 seed each, ~10-15 min on XPU per variant.

Prints a ranking by capacity_avg with HOF/RLF tie-breaks at the end.
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
    "baseline_refresh",
    "lr_1e-3",
    "lr_3e-4",
    "tau_0005",
    "tau_005",
    "hard_update",
]

CFG_DIR = "configs/hp_refresh"
KEY_METRICS = (
    "capacity_avg", "hof_rate", "rlf_rate",
    "ho_rate", "pp_rate", "reliability_pct",
)


def find_run(description: str) -> str | None:
    pattern = os.path.join("results", "benchmarks", f"bmk_*_{description}")
    return sorted(glob.glob(pattern))[-1] if glob.glob(pattern) else None


def read_summary(run_dir: str, seed: int = 42) -> dict[str, float]:
    csv_path = os.path.join(run_dir, "eval", f"qrdqn_summary_seed{seed}.csv")
    if not os.path.exists(csv_path):
        return {}
    out: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"].strip()] = float(row["mean"])
            except (ValueError, KeyError):
                continue
    return out


def run_one(variant: str, device: str) -> tuple[float, str | None]:
    cfg = os.path.join(CFG_DIR, f"{variant}.yaml")
    cmd = [
        "venv-RL/bin/python3", "src/main.py",
        "--config", cfg,
        "--agents", "qrdqn",
        "--device", device,
        "--description", f"hpr_{variant}",
    ]
    print(f"\n>>> {variant}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    return elapsed, find_run(f"hpr_{variant}")


def rank(rows: list[tuple[str, dict[str, float]]]) -> list[tuple[str, dict[str, float]]]:
    def key(item: tuple[str, dict[str, float]]) -> tuple[float, float, float]:
        m = item[1]
        return (
            -m.get("capacity_avg", -1e9),
            m.get("hof_rate", 1e9),
            m.get("rlf_rate", 1e9),
        )
    return sorted([r for r in rows if r[1]], key=key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of variant names.")
    args = parser.parse_args()

    variants = CONFIGS if args.only is None else [
        v.strip() for v in args.only.split(",") if v.strip()
    ]

    print(f"=== HP refresh sweep ===")
    print(f"Variants: {variants}")
    print(f"Device:   {args.device}")

    rows: list[tuple[str, dict[str, float]]] = []
    for v in variants:
        try:
            dur, run_dir = run_one(v, args.device)
        except subprocess.CalledProcessError as err:
            print(f"!!! {v} failed: {err}", file=sys.stderr)
            rows.append((v, {}))
            continue
        metrics = read_summary(run_dir) if run_dir else {}
        rows.append((v, metrics))
        cap = metrics.get("capacity_avg", float("nan"))
        hof = metrics.get("hof_rate", float("nan"))
        print(f"--- {v} done in {dur/60:.1f}min, cap={cap:.3f}, hof={hof:.3f}")

    ranked = rank(rows)
    print("\n" + "=" * 100)
    print("HP REFRESH — RANKING BY capacity_avg (HOF / RLF tie-breaks)")
    print("=" * 100)
    header = f"{'rank':>5}{'  variant':<22}" + "".join(f"{m:>14}" for m in KEY_METRICS)
    print(header)
    print("-" * len(header))
    for rk, (name, metrics) in enumerate(ranked, start=1):
        cells = "".join(f"{metrics.get(m, float('nan')):>14.4f}" for m in KEY_METRICS)
        print(f"{rk:>5}  {name:<20}{cells}")
    print("=" * 100)

    if ranked:
        winner = ranked[0]
        print(f"\nWinner: {winner[0]} (cap_avg = {winner[1].get('capacity_avg', float('nan')):.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
