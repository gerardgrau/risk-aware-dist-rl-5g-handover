"""Plot CVaR alpha sweep (reward vs hof) for paper figure.

Reads multiple benchmark eval CSVs and plots reward and hof_rate as
function of alpha, plus reference baselines.
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def load_bmk_summary(expdir: str, agent: str = "qrdqn") -> dict[str, tuple[float, float]]:
    """Return {metric: (mean, std)} aggregating across seeds."""
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
    # Variants: (alpha, k_effective, bmk_pattern)
    variants = [
        (0.05, 2,  "cvar_a005_N25"),
        (0.10, 3,  "ct_N25_a010"),
        (0.20, 5,  "cvar_a020_N25"),
        (0.30, 8,  "cvar_a030_N25"),
        (0.50, 13, "cvar_a050_N25"),
    ]
    rows = []
    for alpha, k, pat in variants:
        bmk = find_bmk(pat)
        if not bmk:
            print(f"missing: {pat}")
            continue
        m = load_bmk_summary(bmk)
        if not m:
            continue
        rows.append({
            "alpha": alpha, "k": k, "bmk": os.path.basename(bmk),
            "reward_mean": m["reward"][0], "reward_std": m["reward"][1],
            "hof_mean": m["hof_rate"][0], "hof_std": m["hof_rate"][1],
            "cap_mean": m["capacity_avg"][0], "cap_std": m["capacity_avg"][1],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Reference baselines (3-seed)
    dqn_baseline_reward = 1056.4
    dqn_baseline_hof = 2.40
    mid_baseline_reward = 1050.1
    mid_baseline_hof = 2.60

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(df["alpha"], df["reward_mean"], yerr=df["reward_std"],
                marker='o', linewidth=2, markersize=8, capsize=4,
                label='cvar_truncated (N=25, κ=1.0)')
    ax.axhline(dqn_baseline_reward, color='red', linestyle='--', alpha=0.7,
               label=f'DQN baseline ({dqn_baseline_reward})')
    ax.axhline(mid_baseline_reward, color='gray', linestyle='--', alpha=0.7,
               label=f'midpoint baseline ({mid_baseline_reward})')
    ax.set_xlabel('CVaR α (risk fraction)')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward vs CVaR α')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(df["alpha"], df["hof_mean"], yerr=df["hof_std"],
                marker='o', linewidth=2, markersize=8, capsize=4,
                color='C1', label='cvar_truncated (N=25, κ=1.0)')
    ax.axhline(dqn_baseline_hof, color='red', linestyle='--', alpha=0.7,
               label=f'DQN baseline ({dqn_baseline_hof})')
    ax.axhline(mid_baseline_hof, color='gray', linestyle='--', alpha=0.7,
               label=f'midpoint baseline ({mid_baseline_hof})')
    ax.set_xlabel('CVaR α (risk fraction)')
    ax.set_ylabel('HOF Rate (/min)')
    ax.set_title('Handover-Failure Rate vs CVaR α')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('CVaR Risk Fraction Sweep (cvar_truncated, N=25, v2 physics)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = 'results/final_metrics/plots/cvar_alpha_sweep_2026-05-24.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
