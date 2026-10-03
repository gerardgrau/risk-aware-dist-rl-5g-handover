"""Sweep QR-DQN CVaR risk fractions on top of the quantile-ablation base.

Loads `configs/test-quantiles.yaml`, overrides `num_quantiles`, `risk_type`
and `risk_fraction` for each k in the sweep, writes a temp YAML and invokes
`src/main.py`. Aborts on the first failed sub-run so broken sweeps don't
get logged as success.
"""

import argparse
import os
import subprocess
import time

import yaml


def run_cvar_study(device: str) -> None:
    risk_fractions = [0.05, 0.1, 0.25, 0.5]
    config_path = "configs/test-quantiles.yaml"
    N = 200  # Optimal N from the quantile ablation.

    print(f"=== Starting Risk-Management (CVaR) Study ===")
    print(f"Quantiles (N): {N}")
    print(f"Risk Fractions: {risk_fractions}")
    print(f"Device: {device}")

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    results_summary: list[tuple[float, float]] = []

    for k in risk_fractions:
        print(f"\n>>> Running QRDQN with CVaR fraction={k}...")
        temp_config = f"configs/temp_cvar_k{k}.yaml"

        cfg = yaml.safe_load(yaml.safe_dump(base_config))  # deep copy
        cfg.setdefault("agent", {})
        cfg["agent"]["num_quantiles"] = N
        cfg["agent"]["risk_type"] = "cvar"
        cfg["agent"]["risk_fraction"] = k

        with open(temp_config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = [
            "venv-RL/bin/python3", "src/main.py",
            "--config", temp_config,
            "--description", f"cvar-study-k{k}",
            "--device", device,
            "--agents", "qrdqn",
        ]

        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
        finally:
            os.remove(temp_config)
        duration = time.time() - start_time

        results_summary.append((k, duration))
        print(f"--- k={k} completed in {duration:.2f} seconds ---")

    print("\n" + "=" * 40)
    print("      CVaR STUDY SUMMARY")
    print("=" * 40)
    print(f"{'Risk Fraction (k)':<20} | {'Duration (s)':<15}")
    print("-" * 38)
    for k, dur in results_summary:
        print(f"{k:<20} | {dur:<15.2f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    args = parser.parse_args()
    run_cvar_study(args.device)
