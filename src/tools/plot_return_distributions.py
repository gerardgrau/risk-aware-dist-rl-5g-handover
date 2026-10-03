"""Does the distributional agent *see* the risk?

Loads the trained RN model (25-quantile midpoint QR-DQN) and rolls it out
greedily over a handful of UEs, recording, at every decision, the predicted
return distribution of the chosen action together with the serving cell's
moving-average SINR (the state's risk proxy: low SINR = about to lose the
link). It then shows:

  (left)  the predicted return-distribution CDF for a representative *safe*
          (high-SINR) vs *risky* (low-SINR) state -- risky should be lower and
          wider.
  (right) the predicted distribution's inter-decile spread (q90 - q10) binned
          by serving SINR -- if the agent perceives risk, spread rises as SINR
          falls.

Run (PYTHONPATH must include src):
    ./venv-RL/bin/python3 src/tools/plot_return_distributions.py
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config

_CONFIG = "configs/masked/no_gate/finals_n25/rn.yaml"
_NBS = 21
_N_UE = 18  # episodes (UEs) to roll out
# Serving SINR lives in obs[2 + 3*NBS : 2 + 4*NBS], normalised as (snir-15)/25.
_SNIR_OFF = 2 + 3 * _NBS


def _serving_sinr(obs: np.ndarray) -> float:
    serving = int(np.argmax(obs[2 : 2 + _NBS]))
    return float(obs[_SNIR_OFF + serving]) * 25.0 + 15.0


def main() -> None:
    Config.set_config_path(_CONFIG)
    config = Config.get()
    env = LTMEnv(config=config)
    agent = QRDQNAgent(
        config["agent"], env.observation_space, env.action_space
    )
    model = sorted(glob.glob(
        "results/benchmarks/*nogate_final_rn/models/qrdqn_best.pth"))[-1]
    agent.load(model)
    tau = agent.scheme.tau.cpu().numpy()  # 25 quantile fractions

    sinrs: list[float] = []
    dists: list[np.ndarray] = []
    for ue in range(_N_UE):
        obs, _ = env.reset(seed=ue)
        done = False
        while not done:
            mask = env.valid_action_mask()
            with torch.no_grad():
                pred = agent.q_net(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                )[0]  # [A, 25]
                qv = agent.scheme.expectation(pred)
                qv = qv.masked_fill(
                    ~torch.as_tensor(mask, dtype=torch.bool), float("-inf"))
                a = int(qv.argmax())
                dist = pred[a].cpu().numpy()
            sinrs.append(_serving_sinr(obs))
            dists.append(np.sort(dist))
            obs, _, done, _, _ = env.step(a)
    env.close()

    sinrs = np.array(sinrs)
    dists = np.array(dists)  # [N, 25]
    mean_pred = dists.mean(axis=1)
    spread = dists[:, 22] - dists[:, 2]  # ~q90 - q10
    cv = spread / np.maximum(mean_pred, 1e-6)  # relative dispersion
    r_mean = np.corrcoef(sinrs, mean_pred)[0, 1]
    print(f"collected {len(sinrs)} decision states over {_N_UE} UEs")

    # Representative safe / risky states: mean distribution within each SINR
    # tail (NOT re-sorted -- these are already sorted quantile vectors).
    lo_idx = np.where(sinrs <= np.quantile(sinrs, 0.10))[0]
    hi_idx = np.where(sinrs >= np.quantile(sinrs, 0.90))[0]
    safe = dists[hi_idx].mean(axis=0)
    risky = dists[lo_idx].mean(axis=0)
    cv_lo, cv_hi = cv[lo_idx].mean(), cv[hi_idx].mean()

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel A: the predicted return distribution shifts down with link quality.
    axl.step(safe, tau, where="post", color="#2ca02c", lw=2,
             label=f"safe  (SINR≥{np.quantile(sinrs,0.9):.0f} dB),  CV={cv_hi:.2f}")
    axl.step(risky, tau, where="post", color="#d62728", lw=2,
             label=f"risky (SINR≤{np.quantile(sinrs,0.1):.0f} dB),  CV={cv_lo:.2f}")
    axl.set_xlabel("predicted return  F$^{-1}$(τ)")
    axl.set_ylabel("quantile fraction τ")
    axl.set_title("Predicted return distribution (mean over tail states)")
    axl.grid(True, alpha=0.25)
    axl.legend(fontsize=9, loc="lower right")

    # Panel B: relative dispersion (CV) rises as the link degrades -- the
    # regime where risk-aware action selection actually changes decisions.
    bins = np.linspace(np.quantile(sinrs, 0.02), np.quantile(sinrs, 0.98), 11)
    idx = np.digitize(sinrs, bins)
    cx, cy, ce = [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 30:
            cx.append(0.5 * (bins[b - 1] + bins[b]))
            cy.append(cv[m].mean())
            ce.append(cv[m].std() / np.sqrt(m.sum()))
    axr.errorbar(cx, cy, yerr=ce, marker="o", color="#1f77b4", capsize=3)
    axr.set_xlabel("serving-cell SINR (dB)")
    axr.set_ylabel("relative dispersion  (q90 − q10) / mean")
    axr.set_title("Relative return uncertainty rises as the link degrades")
    axr.grid(True, alpha=0.25)

    fig.suptitle(
        "RN agent's learned return distribution vs link quality  "
        f"(corr[SINR, E-return] = {r_mean:+.2f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "results/final_metrics/plots/return_distributions.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    print(f"corr(SINR, E-return)={r_mean:+.3f}  "
          f"rel-dispersion: risky {cv_lo:.3f} vs safe {cv_hi:.3f}")


if __name__ == "__main__":
    main()
