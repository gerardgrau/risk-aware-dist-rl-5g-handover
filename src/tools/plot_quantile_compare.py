"""Side-by-side quantile distribution comparison: midpoint baseline vs cvar_trunc.

Loads two trained agents (DQN-equivalent midpoint baseline + cvar_truncated V3),
samples the same state, and plots the predicted return distribution for the
top-k actions side-by-side. Demonstrates how the risk-aware policy
concentrates capacity on the bottom α=0.5 of the distribution.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config


def _load_agent(config_path: str, model_path: str) -> tuple[QRDQNAgent, LTMEnv]:
    Config.set_config_path(config_path)
    env = LTMEnv(config=Config.get())
    agent = QRDQNAgent(
        Config.get()["agent"], env.observation_space, env.action_space, device="cpu",
    )
    agent.load(model_path)
    return agent, env


def _predict(agent: QRDQNAgent, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (full_distribution[A,Nt], tau_full[Nt], q_values[A])."""
    st = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        predicted = agent.q_net(st)
        full = agent.scheme.assemble_full(predicted)[0].cpu().numpy()
        if agent.risk_type == "cvar":
            q = agent.scheme.cvar(predicted)[0].cpu().numpy()
        else:
            q = agent.scheme.expectation(predicted)[0].cpu().numpy()
    if agent.scheme.has_fixed_endpoints:
        n = agent.scheme.mean_weights.numel()
        tau_full = np.linspace(0.0, 1.0, n)
    else:
        tau_full = agent.scheme.tau.cpu().numpy()
    return full, tau_full, q


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--midpoint-config", default="configs/baselines_v2.yaml",
        help="Config for midpoint baseline",
    )
    p.add_argument(
        "--midpoint-model",
        default="results/benchmarks/bmk_2026-05-23_10_v2_baselines/models/qrdqn_best.pth",
        help="Path to midpoint baseline best model",
    )
    p.add_argument(
        "--cvar-config", default="configs/headline/headline_qrdqn_v3_tf1.yaml",
        help="Config for cvar_truncated V3",
    )
    p.add_argument(
        "--cvar-model",
        default="results/benchmarks/bmk_2026-05-24_36_HEADLINE_v3_qrdqn_h256_a050_k025_tf1/models/qrdqn_best.pth",
        help="Path to cvar_truncated V3 best model",
    )
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--ue-idx", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument(
        "--out",
        default="results/final_metrics/plots/quantile_compare_v3_2026-05-24.png",
    )
    args = p.parse_args()

    # Load midpoint first (its config sets the env). Cvar's config has the
    # same env physics so the env state is comparable.
    mid_agent, mid_env = _load_agent(args.midpoint_config, args.midpoint_model)
    cvar_agent, cvar_env = _load_agent(args.cvar_config, args.cvar_model)

    # Step both envs in parallel from a fixed UE so we sample the same state.
    mid_env.current_ue_idx = args.ue_idx
    cvar_env.current_ue_idx = args.ue_idx
    state_mid, _ = mid_env.reset()
    state_cvar, _ = cvar_env.reset()
    for _ in range(args.n_steps):
        a_mid = mid_agent.select_action(state_mid, 0)
        state_mid, _, done_m, _, _ = mid_env.step(a_mid)
        a_cvar = cvar_agent.select_action(state_cvar, 0)
        state_cvar, _, done_c, _, _ = cvar_env.step(a_cvar)
        if done_m or done_c:
            break

    # Predict on whatever state midpoint ended at (closest to comparable).
    state = state_mid
    full_m, tau_m, q_m = _predict(mid_agent, state)
    full_c, tau_c, q_c = _predict(cvar_agent, state)

    chosen_m = int(np.argmax(q_m))
    chosen_c = int(np.argmax(q_c))
    top_m = np.argsort(-q_m)[:args.top_k].tolist()
    top_c = np.argsort(-q_c)[:args.top_k].tolist()
    if chosen_m not in top_m: top_m[-1] = chosen_m
    if chosen_c not in top_c: top_c[-1] = chosen_c

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    palette = plt.cm.tab10.colors

    ax = axes[0]
    for rank, i in enumerate(top_m):
        color = palette[rank % len(palette)]
        lab = f"BS{i} (Q={q_m[i]:.2f})"
        if i == chosen_m: lab = "★ " + lab
        ax.plot(tau_m, full_m[i], "o-", ms=4, color=color, label=lab)
        ax.axhline(q_m[i], color=color, alpha=0.2, linestyle="--")
    ax.set_xlabel(r"Quantile fraction $\tau$")
    ax.set_ylabel(r"Predicted return $F^{-1}(\tau)$")
    ax.set_title(
        f"Midpoint baseline (N=25)\n"
        f"Network predicts full inverse-CDF; chooses MEAN of distribution"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.02, 1.02)

    ax = axes[1]
    for rank, i in enumerate(top_c):
        color = palette[rank % len(palette)]
        lab = f"BS{i} (CVaR={q_c[i]:.2f})"
        if i == chosen_c: lab = "★ " + lab
        ax.plot(tau_c, full_c[i], "o-", ms=6, color=color, label=lab)
        ax.axhline(q_c[i], color=color, alpha=0.2, linestyle="--")
    ax.axvspan(0.0, cvar_agent.risk_fraction, color="red", alpha=0.08,
               label=f"CVaR mass (α={cvar_agent.risk_fraction})")
    ax.axvline(cvar_agent.risk_fraction, color="red", alpha=0.4, linestyle=":")
    ax.set_xlabel(r"Quantile fraction $\tau$")
    ax.set_title(
        f"cvar_truncated V3 (N=25, α=0.5, k={cvar_agent.scheme.num_predicted})\n"
        f"Network ONLY predicts the bottom α — entire head focused on the risky tail"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.02, 1.02)

    fig.suptitle(
        f"Return-distribution comparison (UE {args.ue_idx}, step {args.n_steps})  —  "
        f"midpoint vs cvar_truncated V3",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"  midpoint chose BS{chosen_m}, Q={q_m[chosen_m]:.3f}")
    print(f"  cvar     chose BS{chosen_c}, CVaR={q_c[chosen_c]:.3f}")


if __name__ == "__main__":
    main()
