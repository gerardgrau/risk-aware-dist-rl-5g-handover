"""Return-distribution DENSITY (PDF) view of a QR-DQN agent's action values.

The companion ``plot_quantiles`` view draws the inverse CDF F^{-1}(tau). Here we
instead show the probability *density* of each action's return: QR-DQN predicts
N quantiles, which are N (near-)equiprobable samples of the return distribution,
so a kernel-density estimate over them gives a smooth, roughly Gaussian curve.
Overlapping the top-k actions makes it obvious which cell the agent prefers and
how much the distributions overlap. A vertical line marks the value the policy
actually ranks by (the mean for a risk-neutral agent, the CVaR for a risk-aware
one) — so you can see the decision criterion sitting on the distribution.

Run from the repo root with PYTHONPATH exported (see CLAUDE.md):

    ./venv-RL/bin/python3 src/tools/plot_return_density.py --ue-idx 500
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.distrl.agents.distributional.qrdqn import QRDQNAgent  # noqa: E402
from src.distrl.envs.ltm_gym import LTMEnv  # noqa: E402
from src.distrl.utils.config import Config  # noqa: E402

_RN = "results/benchmarks/bmk_2026-06-04_11_nogate_final_rn"


def _load_agent(config_path: str, model_path: str) -> tuple[QRDQNAgent, LTMEnv]:
    Config.set_config_path(config_path)
    np.random.seed(0)
    env = LTMEnv(config=Config.get())
    agent = QRDQNAgent(
        Config.get()["agent"], env.observation_space, env.action_space, device="cpu",
    )
    agent.load(model_path)
    return agent, env


def _predict(agent: QRDQNAgent, state: np.ndarray):
    """Return (full[A, N] quantile values, q[A] policy criterion, label)."""
    st = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        predicted = agent.q_net(st)
        full = agent.scheme.assemble_full(predicted)[0].cpu().numpy()
        if agent.risk_type == "cvar":
            q = agent.scheme.cvar(predicted)[0].cpu().numpy()
            crit = f"CVaR({agent.risk_fraction:.2g})"
        else:
            q = agent.scheme.expectation(predicted)[0].cpu().numpy()
            crit = "mean"
    return full, q, crit


def _kde(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """KDE density on `grid`; falls back to a narrow bump if values are flat."""
    if np.std(vals) < 1e-6:
        return np.zeros_like(grid)
    return gaussian_kde(vals, bw_method=0.35)(grid)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=f"{_RN}/config.yaml")
    p.add_argument("--model", default=f"{_RN}/models/qrdqn_best.pth")
    p.add_argument("--label", default="RN (risk-neutral QR-DQN)")
    p.add_argument("--ue-idx", type=int, default=500, help="0-based UE index")
    p.add_argument("--n-steps", type=int, default=120,
                   help="rollout steps to reach a representative decision state")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    agent, env = _load_agent(args.config, args.model)
    env.current_ue_idx = args.ue_idx
    state, _ = env.reset()
    for _ in range(args.n_steps):
        state, _, done, _, _ = env.step(agent.select_action(state, 0))
        if done:
            break

    full, q, crit = _predict(agent, state)
    chosen = int(np.argmax(q))
    top = np.argsort(-q)[: args.top_k].tolist()
    if chosen not in top:
        top[-1] = chosen

    lo = min(full[i].min() for i in top)
    hi = max(full[i].max() for i in top)
    pad = 0.08 * (hi - lo + 1e-6)
    grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = plt.cm.tab10.colors
    for rank, i in enumerate(top):
        color = palette[rank % len(palette)]
        vals = full[i]
        dens = _kde(vals, grid)
        is_chosen = i == chosen
        lab = (f"{'★ ' if is_chosen else ''}BS{i // 3}.{i % 3}  "
               f"({crit}={q[i]:.2f}, mean={vals.mean():.2f})")
        ax.fill_between(grid, dens, color=color, alpha=0.20 if not is_chosen else 0.32)
        ax.plot(grid, dens, color=color, lw=2.4 if is_chosen else 1.6, label=lab)
        # the value the policy ranks by (mean or CVaR) sitting on the density
        ax.axvline(q[i], color=color, ls="--", lw=1.4, alpha=0.8)
        # rug: the underlying quantile samples
        ax.plot(vals, np.full_like(vals, -0.002 * dens.max() - 0.004 * (rank + 1)),
                "|", color=color, ms=6, alpha=0.6)

    ax.set_xlabel("Predicted return")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"Return-distribution density per action — {args.label}\n"
        f"(KDE over the {agent.scheme.num_predicted} predicted quantiles; "
        f"dashed line = the {crit} the policy ranks by; ticks = quantile samples)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, title=f"top-{len(top)} actions")
    ax.grid(True, ls=":", alpha=0.4)
    ax.margins(x=0.01)

    out = args.out or f"figures/03_risk_and_distributions/return_density_ue{args.ue_idx}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"  chose BS{chosen // 3}.{chosen % 3}  ({crit}={q[chosen]:.3f})")


if __name__ == "__main__":
    main()
