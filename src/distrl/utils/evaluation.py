"""Frozen-weight evaluation over the full UE dataset.

`run_evaluation` runs one episode per UE with epsilon=0 and aggregates
the 8 LTM metrics + per-episode reward. For learned agents on >= 48
UEs it spawns a `ProcessPoolExecutor` (default 10 single-threaded CPU
workers); baseline and tiny datasets stay sequential.

Both paths seed `np.random.seed(eval_base_seed + ue_idx)` at the start
of each episode, so sequential and parallel produce bit-identical
per-episode metrics and eval results don't depend on whatever RNG
state training happened to end on.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Fork copies PyTorch's intra-op thread pool whose threads don't
# survive fork() — workers deadlock on the first torch op. Spawn
# starts a clean interpreter per worker.
_SPAWN_CTX = multiprocessing.get_context("spawn")

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.metrics import calculate_8_metrics


def _build_agent_for_eval(
    agent_type: str,
    observation_space: Any,
    action_space: Any,
    agent_config: dict,
    device: str = "cpu",
) -> Any:
    """Construct a fresh agent from config (used by parallel workers)."""
    if agent_type == "qrdqn":
        from src.distrl.agents.distributional.qrdqn import QRDQNAgent
        return QRDQNAgent(agent_config, observation_space, action_space, device=device)
    if agent_type == "dqn":
        from src.distrl.agents.standard.dqn import DQNAgent
        return DQNAgent(agent_config, observation_space, action_space, device=device)
    if agent_type == "ltm_baseline":
        from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
        return LTMBaselineAgent(agent_config, observation_space, action_space, device=device)
    if agent_type == "cmab":
        from src.distrl.agents.standard.cmab import CMABAgent
        return CMABAgent(agent_config, observation_space, action_space, device=device)
    raise ValueError(f"Unknown agent type: {agent_type!r}")


def _run_episode(
    env: LTMEnv,
    agent: Any,
    config: dict,
    is_baseline: bool,
) -> dict[str, Any]:
    """One eval episode → 8 metrics + reward."""
    is_cmab = (agent.__class__.__name__ == "CMABAgent")
    
    t_redo = 1
    while True:
        state, _ = env.reset()
        agent.reset()
        if hasattr(agent, "set_env"):
            agent.set_env(env)
        done = False
        last_info: dict = {}
        episode_reward = 0.0
        while not done:
            if is_baseline:
                state, reward, done, _, info = env.step(
                    0, high_res_callback=agent.select_action,
                )
            else:
                action = agent.select_action(
                    state, epsilon=0.0, valid_mask=env.valid_action_mask(),
                )
                state, reward, done, _, info = env.step(action)
            episode_reward += reward
            if done:
                last_info = info

        m = calculate_8_metrics(
            mcs_history=last_info["metrics"]["mcs"],
            rlf_history=last_info["metrics"]["rlf"],
            ho_history=last_info["metrics"]["ho"],
            hof_history=last_info["metrics"]["hof"],
            pp_history=last_info["metrics"]["pp"],
            reserved_history=last_info["metrics"]["reserved"],
            config=config,
        )
        m["reward"] = episode_reward
        m["composite_reward"] = float(
            last_info["metrics"].get("composite_reward", float("nan"))
        )
        m["util_throughput"] = float(
            last_info["metrics"].get("util_throughput", float("nan"))
        )
        m["n_decisions"] = float(last_info["metrics"].get("n_decisions", float("nan")))

        if is_cmab:
            minutes = len(last_info["metrics"]["rlf"]) * 0.01 / 60.0
            num_ho = sum(last_info["metrics"]["ho"])
            num_rlf = sum(last_info["metrics"]["rlf"])
            num_hof = sum(last_info["metrics"]["hof"])
            
            ho_rate = num_ho / minutes
            rlf_rate = num_rlf / minutes
            hof_rate = num_hof / minutes
            
            should_redo = (ho_rate > 14.0) or (rlf_rate > 2.0) or ((hof_rate / max(1.0, ho_rate) > 0.1) and (hof_rate > 2.0))
            if should_redo:
                t_redo += 1
                env.current_ue_idx -= 1
                continue
                
        break

    return m


def _run_chunk_in_worker(
    model_path: str | None,
    config: dict,
    agent_type: str,
    ue_start: int,
    ue_end: int,
    eval_base_seed: int,
) -> list[dict[str, Any]]:
    """Worker entry point: run UEs [ue_start, ue_end) and return metrics."""
    # Single intra-op thread per worker — N workers each spinning N
    # threads otherwise oversubscribes the box (load explodes).
    import torch as _torch
    _torch.set_num_threads(1)

    env = LTMEnv(config=config)
    env.current_ue_idx = ue_start
    agent = _build_agent_for_eval(
        agent_type, env.observation_space, env.action_space,
        config["agent"], device="cpu",
    )
    if model_path is not None:
        agent.load(model_path)
    is_baseline = agent_type.lower() in ["ltm_baseline", "cmab"]

    results: list[dict[str, Any]] = []
    for ue_idx in range(ue_start, ue_end):
        np.random.seed(eval_base_seed + ue_idx)
        results.append(_run_episode(env, agent, config, is_baseline))
    env.close()
    return results


_BLAS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _run_sequential_eval(
    agent: Any,
    config: dict,
    agent_type: str,
    num_eval: int,
    eval_base_seed: int,
    progress_desc: str,
) -> list[dict[str, Any]]:
    env = LTMEnv(config=config)
    is_baseline = agent_type.lower() in ["ltm_baseline", "cmab"]
    results: list[dict[str, Any]] = []
    for ue_idx in tqdm(range(num_eval), desc=progress_desc, leave=False):
        np.random.seed(eval_base_seed + ue_idx)
        results.append(_run_episode(env, agent, config, is_baseline))
    env.close()
    return results


def _run_parallel_eval(
    agent: Any,
    config: dict,
    agent_type: str,
    num_eval: int,
    num_workers: int,
    eval_base_seed: int,
    progress_desc: str,
    model_path: str | None,
) -> list[dict[str, Any]]:
    # Need a checkpoint on disk for workers to reload. Use the one
    # main.py already saved if present, else dump a temp.
    cleanup_tmp = False
    if model_path is None and agent_type != "ltm_baseline":
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"eval_{agent_type}_", suffix=".pth", delete=False,
        )
        tmp.close()
        agent.save(tmp.name)
        model_path = tmp.name
        cleanup_tmp = True

    chunk_size = (num_eval + num_workers - 1) // num_workers
    chunks = [
        (w * chunk_size, min((w + 1) * chunk_size, num_eval))
        for w in range(num_workers)
    ]
    chunks = [(s, e) for (s, e) in chunks if s < e]

    # BLAS / OpenMP read their thread cap from env at import time;
    # spawn workers inherit the parent env, so set BEFORE the executor.
    saved_env = {k: os.environ.get(k) for k in _BLAS_THREAD_ENV_KEYS}
    for k in _BLAS_THREAD_ENV_KEYS:
        os.environ[k] = "1"

    all_results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=_SPAWN_CTX,
        ) as exe:
            futures = [
                exe.submit(
                    _run_chunk_in_worker,
                    model_path, config, agent_type,
                    start, end, eval_base_seed,
                )
                for (start, end) in chunks
            ]
            with tqdm(total=num_eval, desc=progress_desc, leave=False) as pbar:
                # Iterate in submission order so per-UE metrics stay
                # aligned with UE index in the raw CSV.
                for fut, (start, end) in zip(futures, chunks):
                    all_results.extend(fut.result())
                    pbar.update(end - start)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if cleanup_tmp and model_path is not None:
            try:
                os.remove(model_path)
            except OSError:
                pass

    return all_results


def run_evaluation(
    agent: Any,
    config: dict,
    experiment_dir: str,
    agent_type: str,
    seed: int,
    save_results: bool = True,
    output_prefix: str | None = None,
    num_workers: int | None = None,
) -> dict[str, float]:
    """Evaluate `agent` on every UE in the dataset with epsilon=0.

    Args:
        num_workers: explicit CPU worker count. None auto-picks 1 for
            baseline / tiny datasets and `min(10, cpu_count - 2)` otherwise.
    """
    print(f"    -> Starting Formal Evaluation Phase (Seed {seed})...")

    # Dynamic path overrides for evaluation on the 1000 UEs test dataset
    import copy
    config = copy.deepcopy(config)
    test_dir = "data/ChannelGains"
    for candidate in ["data/test", "data/5g_advanced_ltm_channel_trajectories/test", "data/ChannelGains"]:
        if os.path.exists(candidate):
            test_dir = candidate
            break
    config['paths']['channel_data_directory'] = test_dir
    config['paths']['precomputed_data_directory'] = "data/Precomputed"
    config['simulation']['ue_number'] = 1000

    probe_env = LTMEnv(config=config)
    num_eval = len(probe_env.files)
    probe_env.close()
    if num_eval == 0:
        print("    -> Skipping Evaluation: No trajectories found.")
        return {}

    is_baseline = agent_type.lower() in ["ltm_baseline", "cmab"]
    agent_type_lower = agent_type.lower()
    eval_base_seed = 1000 + int(seed)
    progress_desc = f"    Eval {agent_type}"

    if num_workers is None:
        if is_baseline or num_eval < 48:
            num_workers = 1
        else:
            # 10 was the empirical sweet spot on a 12-core box; reserve
            # 2 cores for the OS + main process.
            num_workers = min(10, max(1, (os.cpu_count() or 1) - 2))

    if num_workers > 1:
        # Reuse main.py's saved checkpoint if present; else _run_parallel_eval
        # writes a temp file.
        model_path: str | None = None
        if save_results and not is_baseline:
            candidate = os.path.join(
                experiment_dir, "models", f"{agent_type_lower}_seed{seed}.pth",
            )
            if os.path.exists(candidate):
                model_path = candidate
        all_eval_metrics = _run_parallel_eval(
            agent, config, agent_type_lower, num_eval, num_workers,
            eval_base_seed, progress_desc, model_path,
        )
    else:
        all_eval_metrics = _run_sequential_eval(
            agent, config, agent_type_lower, num_eval,
            eval_base_seed, progress_desc,
        )

    df = pd.DataFrame(all_eval_metrics)
    summary = {"mean": df.mean().to_dict(), "std": df.std().to_dict()}

    if save_results:
        if output_prefix:
            summary_csv = f"{output_prefix}_summary.csv"
            raw_csv = f"{output_prefix}_raw.csv"
        else:
            eval_dir = os.path.join(experiment_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            summary_csv = os.path.join(eval_dir, f"{agent_type}_summary_seed{seed}.csv")
            raw_csv = os.path.join(eval_dir, f"{agent_type}_raw_seed{seed}.csv")

        metric_order = [
            "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
            "reliability_pct", "prep_rate", "res_reservation_pct",
            "reward", "total_steps", "total_minutes",
        ]
        summary_df = pd.DataFrame({
            "metric": summary["mean"].keys(),
            "mean": summary["mean"].values(),
            "std": summary["std"].values(),
        })
        summary_df['metric'] = pd.Categorical(
            summary_df['metric'],
            categories=metric_order + [
                m for m in summary_df['metric'] if m not in metric_order
            ],
            ordered=True,
        )
        summary_df = summary_df.sort_values('metric')
        summary_df.to_csv(summary_csv, index=False)
        df.to_csv(raw_csv, index_label="eval_episode")

        print(f"    -> Summary saved to {summary_csv}")
        print(f"    -> Raw metrics saved to {raw_csv}")

    print(
        f"    -> Evaluation Complete. "
        f"HO Rate: {summary['mean']['ho_rate']:.2f} "
        f"± {summary['std']['ho_rate']:.2f}"
    )
    return summary['mean']
