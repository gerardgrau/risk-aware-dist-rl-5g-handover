"""Master Orchestration Runner for 60+ Hours Sweep.

Executes all phases of replication and hyperparameter search sequentially,
logs performance to a unified markdown report, and automatically appends
further combinations to run indefinitely if the queue is exhausted.
"""

import os
import sys
import time
import json
import yaml
import glob
import subprocess
import traceback

def run_cmd(cmd):
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR]: Command failed with exit code {result.returncode}")
        print(result.stderr)
    return result

def get_latest_benchmark_dir():
    bmk_dirs = glob.glob("results/benchmarks/bmk_*")
    if not bmk_dirs:
        return None
    return max(bmk_dirs, key=os.path.getmtime)

def append_to_report(desc, config_path, bmk_dir):
    report_path = "results/final_metrics/overnight_massive_sweep_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Initialize report if not exists
    if not os.path.exists(report_path):
        with open(report_path, "w") as f:
            f.write("# Overnight Massive Replications and Hyperparameter Search Sweep Report\n\n")
            f.write("| Configuration | Metrics Path | Reward/Decision | Capacity | Reliability (%) | HO Rate (/min) | HOF Rate (/min) | PP Rate (/min) | RLF Rate (/min) | Prep Rate | Res. Reservation (%) |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    if not bmk_dir:
        return

    # Run headline_to_summary to average all seeds
    agent_name = "qrdqn"
    with open(os.path.join(bmk_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("agent", {}).get("type") == "dqn":
        agent_name = "dqn"
    elif cfg.get("agent", {}).get("type") == "cmab":
        agent_name = "cmab"

    summary_out = os.path.join(bmk_dir, "eval", f"{agent_name}_seed_aggregated.csv")
    cmd = [
        "venv-RL/bin/python3", "src/tools/headline_to_summary.py",
        bmk_dir, agent_name, summary_out
    ]
    run_cmd(cmd)

    if not os.path.exists(summary_out):
        print(f"[WARNING]: Summary CSV not found at {summary_out}")
        return

    # Parse metrics from summary
    try:
        import pandas as pd
        df = pd.read_csv(summary_out).set_index("metric")
        
        m_mean = df["mean"].to_dict()
        m_std = df["std"].to_dict()
        
        # Calculate Reward per decision
        reward_val = m_mean.get("reward", 0.0)
        n_decisions = m_mean.get("n_decisions", 1.0)
        if reward_val > 50.0:
            reward_val = reward_val / max(n_decisions, 1.0)
            reward_std = m_std.get("reward", 0.0) / max(n_decisions, 1.0)
        else:
            reward_std = m_std.get("reward", 0.0)

        row = (
            f"| {desc} (`{os.path.basename(config_path)}`) "
            f"| `{os.path.basename(bmk_dir)}` "
            f"| {reward_val:.3f}±{reward_std:.3f} "
            f"| {m_mean.get('capacity_avg', 0.0):.3f} "
            f"| {m_mean.get('reliability_pct', 0.0):.2f}% "
            f"| {m_mean.get('ho_rate', 0.0):.2f}±{m_std.get('ho_rate', 0.0):.2f} "
            f"| {m_mean.get('hof_rate', 0.0):.2f}±{m_std.get('hof_rate', 0.0):.2f} "
            f"| {m_mean.get('pp_rate', 0.0):.2f}±{m_std.get('pp_rate', 0.0):.2f} "
            f"| {m_mean.get('rlf_rate', 0.0):.3f} "
            f"| {m_mean.get('prep_rate', 0.0):.1f} "
            f"| {m_mean.get('res_reservation_pct', 0.0):.2f}% |\n"
        )
        
        # Check if description is already in the file and replace it to prevent duplicates
        with open(report_path, "r") as f:
            lines = f.read().splitlines()
        
        replaced = False
        new_lines = []
        for line in lines:
            if f"| {desc} (" in line:
                new_lines.append(row.strip())
                replaced = True
            else:
                new_lines.append(line)
        
        if not replaced:
            new_lines.append(row.strip())
            
        with open(report_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[REPORTED]: Metrics written/updated for {desc}")
    except Exception as e:
        print(f"[ERROR]: Failed to write report row: {e}")
        traceback.print_exc()

def main():
    # Base replication queue (Phase 1)
    queue = [
        # description, config_path, agent_type, episodes override (None = use default 6000), overrides
        ("Risk-Neutral QR-DQN N=25 (Width=512)", "configs/masked/no_gate/finals_n25/rn.yaml", "qrdqn", 6000, {"agent": {"hidden_dims": [512, 512]}}),
        ("Soft-step (r=2.0, Width=512)", "configs/masked/no_gate/finals_n25/softcvar_r20.yaml", "qrdqn", 6000, {"agent": {"hidden_dims": [512, 512]}}),
        ("RA-1q (Dedicated Quantile, Width=512)", "configs/masked/no_gate/finals_n25/ra_1q.yaml", "qrdqn", 6000, {"agent": {"hidden_dims": [512, 512]}}),
        
        ("CVaR risk-dial a=0.25 (Width=512)", "configs/masked/no_gate/ablations_n25/cvarfull_a025.yaml", "qrdqn", 6000, {"agent": {"hidden_dims": [512, 512]}}),
        ("CVaR risk-dial a=0.1 (Width=512)", "configs/masked/no_gate/ablations_n25/cvarfull_a010.yaml", "qrdqn", 6000, {"agent": {"hidden_dims": [512, 512]}}),
    ]

    # Infinite Loop Extension logic
    # Keep running by picking the best hyperparameters or running even longer
    infinite_loop_counter = 1

    while True:
        print(f"\n=== Sweep Queue Size: {len(queue)} items ===")
        if not queue:
            print("\n[QUEUE EXHAUSTED]: Generating further exploratory configurations dynamically...")
            # Generate new sweep: train best agents for 12,000 episodes or try combinations of best parameters
            # Let's combine wider layers (512) and best LR (2e-4) or best batch size (128)
            queue.append((
                f"DQN Hybrid Size-LR Limit (12k) (Inf-{infinite_loop_counter})", 
                "configs/masked/no_gate/finals_n25/dqn.yaml", "dqn", 12000, 
                {"agent": {"hidden_dims": [512, 512], "lr": 0.0002}, "_load_checkpoint": "results/benchmarks/bmk_*_dqn-width-512"}
            ))
            queue.append((
                f"RN Hybrid Size-LR Limit (12k) (Inf-{infinite_loop_counter})", 
                "configs/masked/no_gate/finals_n25/rn.yaml", "qrdqn", 12000, 
                {"agent": {"hidden_dims": [512, 512], "lr": 0.0002}, "_load_checkpoint": "results/benchmarks/bmk_*_risk-neutral-qr-dqn-n-25-width-512"}
            ))
            queue.append((
                f"RA-1q Hybrid Size-LR Limit (12k) (Inf-{infinite_loop_counter})", 
                "configs/masked/no_gate/finals_n25/ra_1q.yaml", "qrdqn", 12000, 
                {"agent": {"hidden_dims": [512, 512], "lr": 0.0002}, "_load_checkpoint": "results/benchmarks/bmk_*_ra-1q-dedicated-quantile,-width-512"}
            ))
            queue.append((
                f"DQN Hybrid Size-Batch Limit (12k) (Inf-{infinite_loop_counter})", 
                "configs/masked/no_gate/finals_n25/dqn.yaml", "dqn", 12000, 
                {"agent": {"hidden_dims": [512, 512], "batch_size": 128}, "_load_checkpoint": "results/benchmarks/bmk_*_dqn-width-512"}
            ))
            queue.append((
                f"RN Hybrid Size-Batch Limit (12k) (Inf-{infinite_loop_counter})", 
                "configs/masked/no_gate/finals_n25/rn.yaml", "qrdqn", 12000, 
                {"agent": {"hidden_dims": [512, 512], "batch_size": 128}, "_load_checkpoint": "results/benchmarks/bmk_*_risk-neutral-qr-dqn-n-25-width-512"}
            ))
            
            infinite_loop_counter += 1

        # Dequeue
        item = queue.pop(0)
        desc = item[0]
        config_path = item[1]
        agent_type = item[2]
        episodes = item[3]
        overrides = item[4] if len(item) > 4 else None
        
        # Check if the folder for this run already exists, has evaluation summary, and has matching episodes
        upgrade_checkpoint_dir = None
        desc_normalized = desc.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("=", "-").replace("/", "-")
        existing_dirs = glob.glob(f"results/benchmarks/bmk_*_{desc_normalized}")
        
        is_completed = False
        if existing_dirs:
            # Pick the latest modified folder matching this pattern
            matched_dir = max(existing_dirs, key=os.path.getmtime)
            # Check if eval summary is present (meaning run finished)
            summary_pattern = os.path.join(matched_dir, "eval", "*_summary_seed*.csv")
            if glob.glob(summary_pattern):
                try:
                    with open(os.path.join(matched_dir, "config.yaml"), "r") as f:
                        cfg = yaml.safe_load(f)
                    run_eps = cfg.get("agent", {}).get("num_episodes", 2000)
                except Exception:
                    run_eps = 2000
                
                # If target episodes match or exceed what's requested, skip
                target_eps = episodes if episodes is not None else 6000
                if run_eps >= target_eps:
                    print(f"\n[RESUME]: Found completed folder {matched_dir} for '{desc}' with {run_eps} episodes. Skipping training...")
                    append_to_report(desc, config_path, matched_dir)
                    is_completed = True
                else:
                    print(f"\n[UPGRADE]: Found folder {matched_dir} but it only ran {run_eps} episodes. Rerunning for {target_eps} episodes...")
                    upgrade_checkpoint_dir = matched_dir

        if is_completed:
            continue

        if not is_completed:
            if overrides and "_load_checkpoint" in overrides:
                # Resolve dynamically if it matches a pattern
                cp_pattern = overrides.pop("_load_checkpoint")
                cp_dirs = glob.glob(cp_pattern)
                if cp_dirs:
                    upgrade_checkpoint_dir = max(cp_dirs, key=os.path.getmtime)
                    print(f"\n[LOAD CUSTOM CHECKPOINT]: Explicitly resuming from {upgrade_checkpoint_dir} for '{desc}'")
                else:
                    print(f"\n[WARNING]: Custom checkpoint pattern '{cp_pattern}' not found for '{desc}'! Starting from scratch...")

        # Calculate episodes to run dynamically
        target_eps = episodes if episodes is not None else 6000
        run_eps_to_execute = target_eps
        if upgrade_checkpoint_dir:
            try:
                with open(os.path.join(upgrade_checkpoint_dir, "config.yaml"), "r") as f:
                    cfg = yaml.safe_load(f)
                run_eps = cfg.get("agent", {}).get("num_episodes", 2000)
            except Exception:
                run_eps = 2000
            run_eps_to_execute = max(1, target_eps - run_eps)

        print(f"\n=======================================================")
        print(f" STARTING: {desc}")
        print(f" Config:   {config_path}")
        print(f" Agent:    {agent_type}")
        print(f" Target Episodes: {target_eps}")
        print(f" Episodes to run: {run_eps_to_execute}")
        if upgrade_checkpoint_dir:
            print(f" Resuming from:  {upgrade_checkpoint_dir}")
        if overrides:
            print(f" Overrides: {overrides}")
        print(f"=======================================================")

        # Handle overrides by writing a temporary config file
        target_config = config_path
        if overrides or episodes is not None:
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                
                # Apply nested overrides
                if overrides:
                    for k1, v1 in overrides.items():
                        if isinstance(v1, dict):
                            cfg.setdefault(k1, {})
                            for k2, v2 in v1.items():
                                cfg[k1][k2] = v2
                        else:
                            cfg[k1] = v1
                
                temp_dir = "configs/temp_sweep"
                os.makedirs(temp_dir, exist_ok=True)
                target_config = os.path.join(temp_dir, f"temp_{int(time.time())}.yaml")
                with open(target_config, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            except Exception as e:
                print(f"[ERROR]: Failed to apply overrides: {e}")
                continue

        # Run main.py
        cmd = [
            "venv-RL/bin/python3", "src/main.py",
            "--config", target_config,
            "--agents", agent_type,
            "--device", "cpu",
            "--seeds", "42,43,44,45,46",
            "--description", desc.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("=", "-").replace("/", "-"),
            "--episodes", str(run_eps_to_execute)
        ]
        if upgrade_checkpoint_dir:
            cmd += ["--load-checkpoint", upgrade_checkpoint_dir]

        start_time = time.time()
        try:
            run_cmd(cmd)
        except Exception as e:
            print(f"[CRITICAL EXCEPTION IN TRAINING]: {e}")
            traceback.print_exc()

        duration = time.time() - start_time
        print(f"[COMPLETED]: {desc} in {duration:.2f} seconds")

        # Get latest benchmark directory and write metrics to report
        latest_bmk = get_latest_benchmark_dir()
        append_to_report(desc, config_path, latest_bmk)

        # Cleanup temporary config file if created
        if target_config != config_path and os.path.exists(target_config):
            try:
                os.remove(target_config)
            except Exception as e:
                print(f"[WARNING]: Failed to remove temporary config: {e}")

if __name__ == "__main__":
    main()
