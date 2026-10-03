"""Plot training-reward curves comparing the key headline variants.

Compares:
- DQN baseline_v2
- midpoint baseline (N=25, mean policy)
- cvar_truncated V1 (N=25, α=0.10, κ=1.0, hidden=128)
- cvar_truncated V2 (hidden=256, α=0.50, κ=0.25)
- cvar_truncated V3 (V2 + train_freq=1)

Each plotted as mean ± shaded std across seeds, with a 50-episode rolling
average to denoise.
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_train_curves(expdir: str, agent: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (episode, mean_reward, std_reward) over seeds."""
    train_dir = os.path.join(expdir, "train")
    pattern = os.path.join(train_dir, f"{agent}_*_seed*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return np.array([]), np.array([]), np.array([])
    seed_curves = []
    for p in paths:
        df = pd.read_csv(p)
        # train CSV columns: episode, reward, loss, ...
        if "reward" in df.columns:
            seed_curves.append(df[["episode", "reward"]].set_index("episode"))
        else:
            df.columns = ["episode", "reward"] + list(df.columns[2:])
            seed_curves.append(df[["episode", "reward"]].set_index("episode"))
    combined = pd.concat(seed_curves, axis=1)
    combined.columns = [f"seed_{i}" for i in range(len(seed_curves))]
    mean = combined.mean(axis=1).values
    std = combined.std(axis=1).values
    ep = combined.index.values
    return ep, mean, std


def smooth(y: np.ndarray, w: int = 50) -> np.ndarray:
    if len(y) < w:
        return y
    # Centered rolling mean.
    out = pd.Series(y).rolling(w, min_periods=1, center=True).mean().values
    return out


def find_bmk(pattern: str) -> str | None:
    matches = sorted(
        [p for p in glob.glob("results/benchmarks/bmk_*")
         if os.path.basename(p).endswith(pattern)],
        reverse=True,
    )
    return matches[0] if matches else None


def main() -> None:
    variants = [
        ("DQN baseline V2", "v2_baselines", "dqn", "C3"),
        ("QR-DQN midpoint V2", "v2_baselines", "qrdqn", "C7"),
        ("QR-DQN cvar V1 (h128)", "HEADLINE_qrdqn_ct_a050_k025", "qrdqn", "C0"),
        ("QR-DQN cvar V2 (h256)", "HEADLINE_v2_qrdqn_h256_a050_k025", "qrdqn", "C1"),
        ("QR-DQN cvar V3 (V2+tf=1)", "HEADLINE_v3_qrdqn_h256_a050_k025_tf1", "qrdqn", "C2"),
        ("DQN headline V2 (h256+hardupd)", "HEADLINE_v2_dqn_h256_hardupdate", "dqn", "C4"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, pat, agent, color in variants:
        bmk = find_bmk(pat)
        if not bmk:
            print(f"missing: {pat}")
            continue
        ep, mean, std = load_train_curves(bmk, agent)
        if len(ep) == 0:
            print(f"no train curves for {pat}/{agent}")
            continue
        m = smooth(mean, 50)
        s = smooth(std, 50)
        ax.plot(ep, m, color=color, label=label, linewidth=1.6)
        ax.fill_between(ep, m - s, m + s, color=color, alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (50-ep rolling mean ± std across seeds)")
    ax.set_title("Headline variant training curves (v2 physics, 1000 ep)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    out = "results/final_metrics/plots/headline_curves_2026-05-24.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
