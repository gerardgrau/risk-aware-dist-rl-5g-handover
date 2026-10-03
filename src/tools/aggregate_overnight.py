"""Aggregate the overnight HP-search runs into a single comparison CSV.

Reads `results/benchmarks/bmk_2026-05-19_*/eval/<agent>_summary_seed*.csv`
files (one mean/std row per metric per seed) and emits a flat CSV with one
row per (bmk_dir, agent, seed) and one column per metric.

Run after `scripts/run_overnight_2026-05-19.sh` (or any subset of it)
completes:

    ./venv-RL/bin/python3 src/tools/aggregate_overnight.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

DATE = "2026-05-19"
BENCHMARK_ROOT = Path("results/benchmarks")
OUT_PATH = Path("results/final_metrics/sweeps/overnight_2026-05-19.csv")

# Metric order we care about (matches existing project conventions).
METRICS = [
    "reward",
    "capacity_avg",
    "hof_rate",
    "rlf_rate",
    "ho_rate",
    "pp_rate",
    "reliability_pct",
    "prep_rate",
    "res_reservation_pct",
]


def read_summary(path: Path) -> dict[str, float]:
    """Read a per-seed `<agent>_summary_seed<N>.csv` file (metric,mean,std)."""
    out: dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["metric"]] = float(row["mean"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    bmk_dirs = sorted(BENCHMARK_ROOT.glob(f"bmk_{DATE}_*"))
    if not bmk_dirs:
        print(f"No bmk_{DATE}_* directories found under {BENCHMARK_ROOT}.")
        return

    rows: list[dict[str, str | float]] = []
    seed_pat = re.compile(r"_summary_seed(\d+)\.csv$")

    for bmk in bmk_dirs:
        eval_dir = bmk / "eval"
        if not eval_dir.is_dir():
            continue
        # Description is everything after the date_num_ prefix.
        # bmk_2026-05-19_3_kappa_05 -> "kappa_05"
        desc = "_".join(bmk.name.split("_")[3:])

        for summary in sorted(eval_dir.glob("*_summary_seed*.csv")):
            m = seed_pat.search(summary.name)
            if not m:
                continue
            seed = int(m.group(1))
            agent = summary.name.split("_summary_seed")[0]
            metrics = read_summary(summary)
            row = {
                "bmk": bmk.name,
                "description": desc,
                "agent": agent,
                "seed": seed,
            }
            for k in METRICS:
                row[k] = metrics.get(k, float("nan"))
            rows.append(row)

    if not rows:
        print("Found bmk dirs but no per-seed summaries. Runs in progress?")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bmk", "description", "agent", "seed", *METRICS]
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")

    # Also print a quick reward-ranked summary to stdout.
    print("\nReward ranking (single-seed values; lower-is-worse):")
    print(f"  {'description':<30} {'agent':<8} {'reward':>10} "
          f"{'capacity':>10} {'hof':>8} {'rlf':>8} {'ho':>8}")
    for r in sorted(
        rows, key=lambda x: x.get("reward", float("-inf")), reverse=True
    ):
        print(
            f"  {r['description']:<30} "
            f"{r['agent']:<8} "
            f"{r['reward']:>10.2f} "
            f"{r['capacity_avg']:>10.3f} "
            f"{r['hof_rate']:>8.3f} "
            f"{r['rlf_rate']:>8.3f} "
            f"{r['ho_rate']:>8.3f}"
        )


if __name__ == "__main__":
    main()
