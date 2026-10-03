"""Orchestrator for the 3x3 Soft-Step Sensitivity Grid Sweep.

Sweeps:
  alpha in [0.10, 0.25, 0.50] (risk tail cutoff fraction)
  r in [1.5, 2.0, 3.0]        (tail distortion ratio)

All models are trained for 2,000 episodes on seed 42 with width [512, 512],
N=25 quantiles, and evaluated over all 1,000 unseen test trajectories.
Runs 4 workers in parallel on CPU (capped at 4 threads each to respect 22 vCPUs).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import glob
import json
import os
import subprocess
import sys
import time
import yaml


GRID = [
    (0.10, 1.5),
    (0.10, 2.0),
    (0.10, 3.0),
    (0.25, 1.5),
    (0.25, 2.0),
    (0.25, 3.0),
    (0.50, 1.5),
    (0.50, 2.0),
    (0.50, 3.0),
]


def run_single_job(alpha: float, r: float, repo_root: str, log_file: str) -> dict:
    start_time = time.time()
    tag = f"a{alpha:.2f}_r{r:.1f}"
    desc = f"softstep_grid_{tag}"
    temp_cfg = os.path.join(repo_root, "configs", f"temp_grid_{tag}.yaml")

    # Load champion base config
    base_cfg_path = os.path.join(
        repo_root,
        "results/benchmarks/bmk_2026-07-21_3_soft-step-r-2.0,-width-512/config.yaml"
    )
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply grid parameters
    cfg["agent"]["risk_fraction"] = float(alpha)
    cfg["agent"]["cvar_soft_ratio"] = float(r)
    cfg["agent"]["num_episodes"] = 2000
    cfg["benchmark"]["num_seeds"] = 1
    cfg["simulation"]["ue_number"] = 2000

    with open(temp_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = f"{os.path.join(repo_root, 'src')}:{env.get('PYTHONPATH', '')}"

    cmd = [
        os.path.join(repo_root, "venv-RL/bin/python3"),
        os.path.join(repo_root, "src/main.py"),
        "--config", temp_cfg,
        "--description", desc,
        "--agents", "qrdqn",
        "--seeds", "42",
        "--episodes", "2000",
        "--eval-workers", "4",
        "--device", "cpu",
    ]

    msg = f"[{datetime.now().strftime('%H:%M:%S')}] STARTING job: alpha={alpha:.2f}, r={r:.1f}\n"
    print(msg, end="")
    with open(log_file, "a") as f_log:
        f_log.write(msg)

    proc_log_path = os.path.join(repo_root, "results", f"grid_{tag}.log")
    with open(proc_log_path, "w") as proc_log:
        res = subprocess.run(cmd, cwd=repo_root, env=env, stdout=proc_log, stderr=subprocess.STDOUT)

    duration = time.time() - start_time
    if os.path.exists(temp_cfg):
        try:
            os.remove(temp_cfg)
        except OSError:
            pass

    status = "SUCCESS" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
    msg = f"[{datetime.now().strftime('%H:%M:%S')}] FINISHED job: alpha={alpha:.2f}, r={r:.1f} in {duration/60:.1f}m -> {status}\n"
    print(msg, end="")
    with open(log_file, "a") as f_log:
        f_log.write(msg)

    # Locate generated eval summary
    summary_matches = sorted(glob.glob(os.path.join(repo_root, f"results/benchmarks/*{desc}/eval/qrdqn_summary_seed42.csv")), reverse=True)
    summary_path = summary_matches[0] if summary_matches else None

    return {
        "alpha": alpha,
        "r": r,
        "status": status,
        "duration_seconds": duration,
        "summary_csv": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers (default 4)")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.makedirs(os.path.join(repo_root, "results/final_metrics/sweeps"), exist_ok=True)
    log_file = os.path.join(repo_root, "results/final_metrics/sweeps/softstep_grid_progress.log")

    with open(log_file, "w") as f:
        f.write(f"=== Starting 3x3 Soft-Step Grid Sweep: {datetime.now().isoformat()} ===\n")
        f.write(f"Grid: {len(GRID)} configurations, max_workers={args.workers}\n\n")

    print(f"=== Starting 3x3 Soft-Step Grid Sweep ({len(GRID)} configurations, {args.workers} workers) ===")
    print(f"Master log: {log_file}")

    start_total = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_job, alpha, r, repo_root, log_file): (alpha, r)
            for alpha, r in GRID
        }
        for future in as_completed(futures):
            alpha, r = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                err_msg = f"[{datetime.now().strftime('%H:%M:%S')}] EXCEPTION in job alpha={alpha}, r={r}: {e}\n"
                print(err_msg, end="")
                with open(log_file, "a") as f:
                    f.write(err_msg)
                results.append({"alpha": alpha, "r": r, "status": f"ERROR: {e}", "duration_seconds": 0, "summary_csv": None})

    total_time = time.time() - start_total
    summary_file = os.path.join(repo_root, "results/final_metrics/sweeps/softstep_grid_results.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    msg = f"\n=== All 9 Grid Jobs Completed in {total_time/60:.1f} minutes! Saved summary to {summary_file} ===\n"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg)


if __name__ == "__main__":
    main()
