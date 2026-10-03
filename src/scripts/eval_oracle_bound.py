"""Evaluation of Theoretical Upper Bounds: LTM-Constrained Oracle vs. Genie Bound.

Computes the theoretical throughput ceilings across mobile test trajectories:
1. Unconstrained Genie Bound: non-causal maximum MCS across all 21 sectors.
2. LTM-Constrained Execution Oracle: clairvoyant instantaneous selection of the
   strongest prepared candidate sector (or serving cell) at each transmission tick,
   establishing the protocol-constrained operational capacity upper bound.
3. Compares the resulting bounds against Soft-step QR-DQN and 3GPP heuristics
   as reported in IEEE TMLCN Section VI.D.
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config


def evaluate_oracle_bounds(num_ues: int = 100, config_path: str = "configs/config.yaml") -> None:
    Config.set_config_path(config_path)
    cfg = Config.get()
    env = LTMEnv(config=cfg)

    total_files = len(env.files)
    eval_count = min(num_ues, total_files)
    print(f"Evaluating Oracle Upper Bounds across {eval_count} trajectories (total pool: {total_files})...")

    prep_oracle_caps = []
    genie_caps = []

    for ue_idx in tqdm(range(eval_count), desc="Evaluating Bounds"):
        obs, info = env.reset(options={"user_id": ue_idx})
        done = False
        while not done:
            obs, rew, done, trunc, info = env.step(-1)

        reserved = info["metrics"]["reserved"]
        serving = info["metrics"]["serving"]

        d = np.load(env.files[ue_idx])
        all_mcs = d["all_mcs_episode"]
        if all_mcs.shape[0] != 21:
            all_mcs = all_mcs.T

        # 1. Unconstrained Genie Bound: max over all 21 sectors
        genie_best = np.max(all_mcs, axis=0)
        genie_caps.append(np.mean(genie_best))

        # 2. LTM-Constrained Prepared Oracle: max over prepared sectors + serving
        candidate_mask = (reserved > 0)
        for t in range(len(serving)):
            if serving[t] >= 0:
                candidate_mask[serving[t], t] = True

        masked_mcs = np.where(candidate_mask, all_mcs, -1.0)
        best_candidate_mcs = np.max(masked_mcs, axis=0)
        prep_oracle_caps.append(np.mean(best_candidate_mcs))

    genie_mean, genie_std = np.mean(genie_caps), np.std(genie_caps)
    prep_mean, prep_std = np.mean(prep_oracle_caps), np.std(prep_oracle_caps)

    # Reference numbers from the benchmark (Table II)
    softstep_cap = 4.04
    baseline_cap = 3.75

    pct_softstep = (softstep_cap / prep_mean) * 100.0
    pct_baseline = (baseline_cap / prep_mean) * 100.0
    gap_closed = ((softstep_cap - baseline_cap) / (prep_mean - baseline_cap)) * 100.0

    bandwidth_mhz = 20.0  # 20 MHz carrier

    print("\n" + "=" * 78)
    print(" 5G-ADVANCED LTM THEORETICAL BOUNDS & BENCHMARK COMPARISON (IEEE TMLCN Sec. VI.D)")
    print("=" * 78)
    print(f"{'Method / Bound':<38} {'Spectral Eff.':<18} {'Throughput':<14} {'% of LTM Bound'}")
    print("-" * 78)
    print(f"{'Unconstrained Genie Bound (All 21)':<38} {genie_mean:.2f} ± {genie_std:.2f} bps/Hz    {genie_mean * bandwidth_mhz:.1f} Mbps      {genie_mean / prep_mean * 100:.1f}%")
    print(f"{'LTM-Constrained Oracle (Prepared)':<38} {prep_mean:.2f} ± {prep_std:.2f} bps/Hz    {prep_mean * bandwidth_mhz:.1f} Mbps      100.0% (Ceiling)")
    print("-" * 78)
    print(f"{'Soft-step QR-DQN (Proposed)':<38} {softstep_cap:.2f} ± 0.01 bps/Hz    {softstep_cap * bandwidth_mhz:.1f} Mbps      {pct_softstep:.1f}%")
    print(f"{'3GPP Heuristic Baseline':<38} {baseline_cap:.2f} ± 0.29 bps/Hz    {baseline_cap * bandwidth_mhz:.1f} Mbps      {pct_baseline:.1f}%")
    print("=" * 78)
    print(f"Operational Gap Closed by Soft-step: {gap_closed:.1f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LTM Oracle and Genie bounds.")
    parser.add_argument("--num_ues", type=int, default=100, help="Number of trajectories to evaluate (default: 100)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    evaluate_oracle_bounds(num_ues=args.num_ues, config_path=args.config)
