"""Plot Huber kappa sweep (reward vs hof) for paper figure.

Reads kappa sweep eval CSVs at α=0.50 N=25 (cvar_trunc).
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_bmk_summary(expdir: str, agent: str = "qrdqn") -> dict[str, tuple[float, float]]:
    eval_dir = os.path.join(expdir, "eval")
    if not os.path.isdir(eval_dir):
        return {}
    seed_dfs = []
    for path in sorted(glob.glob(os.path.join(eval_dir, f"{agent}_summary_seed*.csv"))):
        seed_dfs.append(pd.read_csv(path).set_index("metric")["mean"])
    if not seed_dfs:
        return {}
    combined = pd.concat(seed_dfs, axis=1)
    return {m: (combined.loc[m].mean(), combined.loc[m].std())
            for m in combined.index}


def find_bmk(pattern: str) -> str | None:
    matches = sorted(glob.glob(f"results/benchmarks/*{pattern}*"), reverse=True)
    return matches[0] if matches else None


def main() -> None:
    # Variants: (kappa, bmk_pattern)
    variants = [
        (0.25, "kappa_025_a050"),
        (0.5,  "kappa_05_a050"),
        (1.0,  "cvar_a050_N25"),   # κ=1.0 baseline at α=0.50
        (2.0,  "kappa_20_a050"),
    ]
    rows = []
    for kappa, pat in variants:
        bmk = find_bmk(pat)
        if not bmk:
            print(f"missing: {pat}")
            continue
        m = load_bmk_summary(bmk)
        if not m:
            continue
        rows.append({
            "kappa": kappa, "bmk": os.path.basename(bmk),
            "reward_mean": m["reward"][0], "reward_std": m["reward"][1],
            "hof_mean": m["hof_rate"][0], "hof_std": m["hof_rate"][1],
        })
    df = pd.DataFrame(rows).sort_values("kappa")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(df["kappa"], df["reward_mean"], yerr=df["reward_std"],
                marker='o', linewidth=2, markersize=8, capsize=4)
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
    ax.set_xlabel('Huber κ (Q-loss threshold)')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward vs Huber κ (cvar_trunc α=0.50, N=25)')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(df["kappa"], df["hof_mean"], yerr=df["hof_std"],
                marker='o', linewidth=2, markersize=8, capsize=4, color='C1')
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
    ax.set_xlabel('Huber κ (Q-loss threshold)')
    ax.set_ylabel('HOF Rate (/min)')
    ax.set_title('Handover-Failure Rate vs Huber κ')
    ax.grid(alpha=0.3)

    fig.suptitle('Huber κ Sweep (cvar_truncated α=0.50, N=25, v2 physics)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = 'results/final_metrics/plots/kappa_sweep_2026-05-24.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
