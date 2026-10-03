"""Per-UE tail analysis: does risk aversion help the *worst* users?

The 8 headline metrics are means over 1,000 UEs, but the whole premise of
risk-aware RL is improving the unlucky tail of the per-user outcome
distribution. This tool pools the per-UE rows from every seed's frozen
evaluation and plots:

  (left)  CCDF of per-UE RLF rate -- P(RLF > x). A shorter/lower tail means
          fewer users suffer radio-link failures.
  (right) CDF of per-UE reward/action -- the left edge is the "unlucky user"
          floor. Risk aversion should lift it; over-correction lowers it.

Run (PYTHONPATH must include src):
    ./venv-RL/bin/python3 src/tools/plot_per_ue_tails.py
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = "results/benchmarks"
# (label, run-folder suffix, agent prefix, colour) in frontier order.
_AGENTS: list[tuple[str, str, str, str]] = [
    ("DQN", "nogate_final_dqn", "dqn", "#555555"),
    ("RN", "nogate_final_rn", "qrdqn", "#1f77b4"),
    ("soft-step r=2", "nogate_final_softcvar_r20", "qrdqn", "#2ca02c"),
    ("RA-1q", "nogate_final_ra_1q", "qrdqn", "#ff7f0e"),
    ("RA (k=7)", "nogate_final_ra", "qrdqn", "#d62728"),
]


def _pool(suffix: str, agent: str) -> dict[str, np.ndarray]:
    """Pool per-UE columns across all seed CSVs for one run."""
    folders = sorted(glob.glob(f"{_BENCH}/*{suffix}"))
    if not folders:
        raise FileNotFoundError(f"no benchmark folder matching *{suffix}")
    rlf, rpa = [], []
    for f in sorted(glob.glob(f"{folders[-1]}/eval/{agent}_raw_seed*.csv")):
        for x in csv.DictReader(open(f)):
            rlf.append(float(x["rlf_rate"]))
            rpa.append(float(x["reward"]) / max(1e-9, float(x["n_decisions"])))
    return {"rlf": np.array(rlf), "rpa": np.array(rpa)}


def _ccdf(ax: plt.Axes, data: np.ndarray, label: str, colour: str) -> None:
    xs = np.sort(data)
    ccdf = 1.0 - np.arange(len(xs)) / len(xs)
    ax.step(xs, ccdf, where="post", label=label, color=colour, lw=1.8)


def _cdf(ax: plt.Axes, data: np.ndarray, label: str, colour: str) -> None:
    xs = np.sort(data)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, cdf, where="post", label=label, color=colour, lw=1.8)


def main() -> None:
    pooled = {lbl: _pool(suf, ag) for lbl, suf, ag, _ in _AGENTS}
    colours = {lbl: c for lbl, _, _, c in _AGENTS}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.3))

    for lbl in pooled:
        _ccdf(axl, pooled[lbl]["rlf"], lbl, colours[lbl])
    axl.set_xlabel("per-UE RLF rate (failures / min)")
    axl.set_ylabel("P(RLF > x)")
    axl.set_title("Radio-link-failure tail (lower = fewer failing users)")
    axl.set_yscale("log")
    axl.set_xlim(left=0.0)
    axl.grid(True, which="both", alpha=0.25)
    axl.legend(fontsize=9)

    for lbl in pooled:
        _cdf(axr, pooled[lbl]["rpa"], lbl, colours[lbl])
    axr.set_xlabel("per-UE reward / action")
    axr.set_ylabel("CDF")
    axr.set_title("Reward floor (left edge = unlucky-user outcome)")
    # Zoom on the lower tail where the floor lives.
    lo = min(np.quantile(pooled[l]["rpa"], 0.005) for l in pooled)
    axr.set_xlim(lo, np.median(pooled["DQN"]["rpa"]) + 0.5)
    axr.set_ylim(0, 0.5)
    axr.grid(True, alpha=0.25)
    axr.legend(fontsize=9)

    fig.suptitle(
        "Per-UE outcome tails (pooled over 5 seeds x 1,000 UEs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "results/final_metrics/plots/per_ue_tails.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")

    # Print the numeric summary that backs the figure.
    print(f"\n{'agent':14}{'meanRLF':>9}{'P(RLF>0)':>9}{'RLFw5%':>8}"
          f"{'maxRLF':>8}{'rpaFloor10%':>12}")
    for lbl in pooled:
        rlf, rpa = pooled[lbl]["rlf"], pooled[lbl]["rpa"]
        w5 = np.sort(rlf)[-max(1, int(0.05 * len(rlf))):].mean()
        floor = np.sort(rpa)[: max(1, int(0.10 * len(rpa)))].mean()
        print(f"{lbl:14}{rlf.mean():9.4f}{(rlf > 0).mean():9.3f}{w5:8.3f}"
              f"{rlf.max():8.2f}{floor:12.3f}")


if __name__ == "__main__":
    main()
