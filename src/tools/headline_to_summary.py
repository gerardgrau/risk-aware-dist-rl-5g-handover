"""Aggregate a multi-seed bmk dir's per-seed eval summaries into one
master summary CSV in the format expected by generate_final_plots.py.

Format: metric, mean (cross-seed mean of per-seed means), std (cross-seed std).
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd


_EVAL_RE = re.compile(r"^(?P<agent>.+)_summary_seed(?P<seed>\d+)\.csv$")


def aggregate(expdir: str, agent: str, out_path: str) -> None:
    eval_dir = os.path.join(expdir, "eval")
    seed_dfs: list[pd.DataFrame] = []
    for path in sorted(glob.glob(os.path.join(eval_dir, f"{agent}_summary_seed*.csv"))):
        if not _EVAL_RE.match(os.path.basename(path)):
            continue
        df = pd.read_csv(path).set_index("metric")
        seed_dfs.append(df["mean"])

    if not seed_dfs:
        print(f"NO {agent} seeds in {eval_dir}")
        return

    combined = pd.concat(seed_dfs, axis=1)
    out = pd.DataFrame({
        "metric": combined.index,
        "mean": combined.mean(axis=1).values,
        "std": combined.std(axis=1).values,
    })
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(seed_dfs)} seeds aggregated)")
    print(out.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("expdir", help="Benchmark dir with eval/")
    p.add_argument("agent", help="Agent name (dqn or qrdqn)")
    p.add_argument("out", help="Output summary CSV path")
    args = p.parse_args()
    aggregate(args.expdir, args.agent, args.out)


if __name__ == "__main__":
    main()
