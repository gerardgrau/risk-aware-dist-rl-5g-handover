"""Plot N (num_quantiles) sweep for midpoint vs cvar_truncated.

Demonstrates:
- midpoint axis is flat (midpoint+mean ≡ DQN)
- cvar_truncated benefits from larger N at α=0.10 up to ~N=100
- past N=200 is diminishing
- matched-K (ct N=250 vs midpoint N=25) shows risk effect is intrinsic
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
    """Find the most recent bmk dir whose name ENDS with pattern.
    Anchored at end to avoid N10 matching N100, etc.
    """
    matches = sorted(
        [p for p in glob.glob("results/benchmarks/bmk_*")
         if os.path.basename(p).endswith(pattern)],
        reverse=True,
    )
    return matches[0] if matches else None


def collect(variants: list[tuple[int, str]]) -> pd.DataFrame:
    rows = []
    for N, pat in variants:
        bmk = find_bmk(pat)
        if not bmk:
            print(f"missing: {pat}")
            continue
        m = load_bmk_summary(bmk)
        if not m:
            continue
        rows.append({
            "N": N,
            "bmk": os.path.basename(bmk),
            "reward_mean": m["reward"][0], "reward_std": m["reward"][1],
            "hof_mean": m["hof_rate"][0], "hof_std": m["hof_rate"][1],
        })
    return pd.DataFrame(rows).sort_values("N")


def main() -> None:
    # midpoint variants: (N, bmk_pattern)
    midpoint_variants = [
        (10,  "n_sweep_mid_N10"),
        (50,  "n_sweep_mid_N50"),
        (100, "n_sweep_mid_N100"),
    ]
    # cvar_trunc variants
    ct_variants = [
        (25,  "ct_N25_a010"),
        (50,  "ct_N50_a010"),
        (100, "ct_N100_a010"),
        (200, "ct_N200_a010"),
        (250, "ct_N250_a010_matchedK"),
        (500, "cvar_trunc_N500_a010"),
    ]
    df_mid = collect(midpoint_variants)
    df_ct = collect(ct_variants)

    # Add the baseline midpoint N=25 from the v2 baseline run (anchor for both)
    base_bmk = find_bmk("v2_baselines")
    base = load_bmk_summary(base_bmk, "qrdqn") if base_bmk else {}
    if base:
        df_mid = pd.concat([
            df_mid,
            pd.DataFrame([{
                "N": 25, "bmk": "baseline",
                "reward_mean": base["reward"][0], "reward_std": base["reward"][1],
                "hof_mean": base["hof_rate"][0], "hof_std": base["hof_rate"][1],
            }]),
        ]).sort_values("N")

    print("midpoint:")
    print(df_mid.to_string(index=False))
    print("\ncvar_trunc:")
    print(df_ct.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.errorbar(df_mid["N"], df_mid["reward_mean"], yerr=df_mid["reward_std"],
                marker='s', label='midpoint (mean policy)', linewidth=2, capsize=4)
    ax.errorbar(df_ct["N"], df_ct["reward_mean"], yerr=df_ct["reward_std"],
                marker='o', label='cvar_truncated α=0.10 (k = ⌈N×α⌉)',
                linewidth=2, capsize=4, color='C2')
    ax.set_xscale('log')
    ax.set_xlabel('Num quantiles (N)')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward vs N')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(df_mid["N"], df_mid["hof_mean"], yerr=df_mid["hof_std"],
                marker='s', label='midpoint (mean policy)', linewidth=2, capsize=4)
    ax.errorbar(df_ct["N"], df_ct["hof_mean"], yerr=df_ct["hof_std"],
                marker='o', label='cvar_truncated α=0.10',
                linewidth=2, capsize=4, color='C2')
    ax.set_xscale('log')
    ax.set_xlabel('Num quantiles (N)')
    ax.set_ylabel('HOF Rate (/min)')
    ax.set_title('HOF Rate vs N (cvar_truncated halves it irrespective of N)')
    ax.legend(loc='center right', fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Num-Quantiles Sweep (v2 physics)', fontsize=13, y=1.02)
    fig.tight_layout()
    out = 'results/final_metrics/plots/n_sweep_2026-05-24.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
