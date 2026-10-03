"""Uniform-random execution policy: the uninformed reward-per-decision floor.

Evaluates a policy that, at every no-gate decision opportunity, chooses
uniformly among the *valid* actions --- the serving cell (stay) plus the
LTM-prepared candidates (`env.valid_action_mask()`) --- i.e. the exact same
masked action set the learned agents select from. It therefore isolates the
value of the execution *decision* while holding LTM's preparation machinery
fixed, giving a scale anchor that is directly comparable to the learned agents
in `tab:final` (Section VI of the paper).

Run over the full 1,000-UE set, in the no-gate (`learned_trigger: true`) regime,
exactly as the learned agents are evaluated:

    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    ./venv-RL/bin/python3 src/scripts/eval_random_policy.py

The reported reward per decision is sum(reward)/sum(n_decisions) over the UEs,
matching `tab:final`; the 8 KPIs are the cross-UE mean +/- std.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

# Ensure the repo root is importable (mirrors the other src/scripts entry points).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config
from src.distrl.utils.metrics import calculate_8_metrics

# Paper KPI order (Reward first, then quality, then mobility/safety, then cost).
_REPORT_ORDER = [
    ("reward_per_decision", "Reward/decision", "{:.4f}"),
    ("capacity_avg", "Capacity (bps/Hz)", "{:.3f}"),
    ("reliability_pct", "Reliability (%)", "{:.2f}"),
    ("ho_rate", "HO/min", "{:.3f}"),
    ("pp_rate", "PP/min", "{:.3f}"),
    ("hof_rate", "HOF/min", "{:.3f}"),
    ("rlf_rate", "RLF/min", "{:.4f}"),
    ("prep_rate", "Prep/min", "{:.2f}"),
    ("res_reservation_pct", "Res. reserv. (%)", "{:.3f}"),
]


def _random_valid_action(mask: np.ndarray) -> int:
    """Pick uniformly among the True entries of the action mask (>= 1 exists)."""
    idx = np.flatnonzero(mask)
    return int(np.random.choice(idx)) if idx.size else 0


def _run_random_episode(env: LTMEnv, config: dict) -> dict[str, float]:
    """One eval episode under the random policy -> 8 KPIs + reward + n_decisions."""
    state, _ = env.reset()
    done = False
    episode_reward = 0.0
    last_info: dict = {}
    while not done:
        action = _random_valid_action(env.valid_action_mask())
        state, reward, done, _, last_info = env.step(action)
        episode_reward += reward

    metrics = calculate_8_metrics(
        mcs_history=last_info["metrics"]["mcs"],
        rlf_history=last_info["metrics"]["rlf"],
        ho_history=last_info["metrics"]["ho"],
        hof_history=last_info["metrics"]["hof"],
        pp_history=last_info["metrics"]["pp"],
        reserved_history=last_info["metrics"]["reserved"],
        config=config,
    )
    metrics["reward"] = episode_reward
    metrics["n_decisions"] = float(last_info["metrics"]["n_decisions"])
    return metrics


def evaluate_random_policy(
    config_path: str, ue_number: int, seed_base: int,
) -> dict[str, tuple[float, float]]:
    """Evaluate the random policy and return {kpi: (mean, std)}.

    Args:
        config_path: YAML config; must be no-gate (`simulation.learned_trigger`).
        ue_number: Number of UEs to run; <= 0 means the full dataset.
        seed_base: Per-UE NumPy seed offset for reproducibility.

    Returns:
        Mapping from KPI name to (mean, std); `reward_per_decision` is the
        sum(reward)/sum(n_decisions) aggregate (its std is left at 0.0).
    """
    Config.set_config_path(config_path)
    config = Config.get()
    if not config.get("simulation", {}).get("learned_trigger", False):
        print(
            "WARNING: 'simulation.learned_trigger' is not true in "
            f"{config_path}; the random floor will be measured in the GATED "
            "regime and is NOT comparable to the no-gate learned agents.",
            file=sys.stderr,
        )

    env = LTMEnv(config=config)
    n_all = len(env.files)
    n = n_all if ue_number <= 0 else min(ue_number, n_all)

    rows: list[dict[str, float]] = []
    tot_reward, tot_decisions = 0.0, 0.0
    for ue_idx in tqdm(range(n), desc="Random policy", leave=False):
        np.random.seed(seed_base + ue_idx)
        m = _run_random_episode(env, config)
        rows.append(m)
        tot_reward += m["reward"]
        tot_decisions += m["n_decisions"]
    env.close()

    kpis = [k for k, _, _ in _REPORT_ORDER if k != "reward_per_decision"]
    out: dict[str, tuple[float, float]] = {
        "reward_per_decision": (tot_reward / tot_decisions, 0.0),
    }
    for k in kpis:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    out["_n_ue"] = (float(n), 0.0)
    out["_mean_decisions"] = (tot_decisions / n, 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/masked/no_gate/finals_n25/rn.yaml",
        help="No-gate (learned_trigger: true) YAML config.",
    )
    parser.add_argument(
        "--ue_number", type=int, default=0,
        help="UEs to evaluate (<=0 = full dataset).",
    )
    parser.add_argument("--seed_base", type=int, default=1042)
    args = parser.parse_args()

    results = evaluate_random_policy(args.config, args.ue_number, args.seed_base)

    n_ue = int(results["_n_ue"][0])
    print(f"\nUniform-random execution policy ({n_ue} UEs, "
          f"{results['_mean_decisions'][0]:.1f} decisions/ep):")
    print("-" * 40)
    for key, label, fmt in _REPORT_ORDER:
        mean, std = results[key]
        if key == "reward_per_decision":
            print(f"  {label:<20s} {fmt.format(mean)}  (sum/sum)")
        else:
            print(f"  {label:<20s} {fmt.format(mean)} +/- {fmt.format(std)}")


if __name__ == "__main__":
    main()
