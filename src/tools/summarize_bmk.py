"""Print one-line summary of a bmk dir for the overnight log.

Reads all eval/<agent>_summary_seed*.csv files and emits:
  <agent>: reward=X.X capacity=X.X reliability=X.X hof=X.X pp=X.X (n seeds)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd


_EVAL_RE = re.compile(r"^(?P<agent>.+)_summary_seed(?P<seed>\d+)\.csv$")
_METRICS = [
    "reward", "capacity_avg", "reliability_pct",
    "ho_rate", "pp_rate", "hof_rate", "rlf_rate",
]


def summarize(expdir: str) -> None:
    eval_dir = os.path.join(expdir, "eval")
    if not os.path.isdir(eval_dir):
        print(f"NO eval dir: {expdir}")
        return

    by_agent: dict[str, list[dict[str, float]]] = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "*_summary_seed*.csv"))):
        m = _EVAL_RE.match(os.path.basename(path))
        if not m:
            continue
        agent = m.group("agent")
        df = pd.read_csv(path)
        vals = {}
        for metric in _METRICS:
            row = df[df["metric"] == metric]
            if not row.empty:
                vals[metric] = float(row.iloc[0]["mean"])
        by_agent.setdefault(agent, []).append(vals)

    print(f"\n=== {os.path.basename(expdir)} ===")
    for agent, seed_vals in by_agent.items():
        if not seed_vals:
            continue
        n = len(seed_vals)
        avg = {m: sum(s.get(m, 0) for s in seed_vals) / n for m in _METRICS}
        std = {
            m: (sum((s.get(m, 0) - avg[m])**2 for s in seed_vals) / n) ** 0.5
            for m in _METRICS
        }
        print(
            f"  {agent} ({n} seeds): "
            f"reward={avg['reward']:.1f}±{std['reward']:.1f} "
            f"cap={avg['capacity_avg']:.3f}±{std['capacity_avg']:.3f} "
            f"rel={avg['reliability_pct']:.2f}±{std['reliability_pct']:.2f} "
            f"hof={avg['hof_rate']:.2f}±{std['hof_rate']:.2f} "
            f"pp={avg['pp_rate']:.2f}±{std['pp_rate']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expdirs", nargs="+", help="Benchmark dir(s) to summarize")
    args = parser.parse_args()
    for d in args.expdirs:
        summarize(d)


if __name__ == "__main__":
    main()
