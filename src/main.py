"""Benchmark entry point.

Single CLI that handles both the canonical multi-seed parallel workflow
and one-off single-seed runs. By default, given `--seeds A,B,C` it forks
one subprocess per seed (all writing to a shared experiment directory),
waits for them, then runs the aggregator. With a single seed (or
`--no-parallel`), runs inline in the same process.

Internal: subprocesses are launched with `--worker-seed N --experiment-dir
PATH` and skip the orchestration / aggregation logic.

Live progress: each subprocess writes its tqdm to
`<expdir>/logs/seed<N>.log`. Use `tail -f` on those for per-seed bars;
the orchestrator only prints start/finish lines.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Ensure src is in PYTHONPATH for absolute imports.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.replay_buffer import ReplayBuffer
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.agents.standard.cmab import CMABAgent
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config
from src.distrl.utils.evaluation import run_evaluation
from src.distrl.utils.metrics import calculate_8_metrics
from src.tools.aggregate import aggregate


class CSVLogger:
    """CSV writer that flushes after every row (so logs survive a
    killed training run). Opens the file in overwrite mode, so reusing
    the same path between runs starts fresh."""

    def __init__(self, filepath: str, headers: list[str]):
        self.filepath = filepath
        self.headers = headers
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self._fh = open(self.filepath, 'w', newline='')
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.headers)
        self._fh.flush()

    def log(self, row: list[Any]):
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _emit_nstep(buf: Any, q: deque, gamma: float) -> None:
    """Pop the oldest transition in q and push its n-step bootstrap to buf.

    The discounted return is summed until either the queue is exhausted
    or a terminal flag is hit (whichever comes first). next_state/done
    come from the last transition included, so a (1-done)*gamma_n
    bootstrap in the agent zeros out correctly at episode end.
    """
    s0, a0, _, _, _ = q[0]
    R = 0.0
    i = 0
    for i, (_, _, r, _, d) in enumerate(q):
        R += (gamma ** i) * r
        if d:
            break
    _, _, _, ns, dn = q[i]
    buf.push(s0, a0, R, ns, dn)
    q.popleft()


def run_seed(
    agent_type: str, env_name: str, seed: int, config: dict,
    experiment_dir: str, pbar: tqdm, device: str = "cpu",
    save_results: bool = True, eval_workers: int | None = None,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if env_name != "ltm":
        raise ValueError(f"Only 'ltm' environment is supported. Received: {env_name}")

    env = LTMEnv(config=config)
    agent_config = config['agent']

    if agent_type.lower() == "dqn":
        agent = DQNAgent(agent_config, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "qrdqn":
        agent = QRDQNAgent(agent_config, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "ltm_baseline":
        agent = LTMBaselineAgent(config, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "cmab":
        agent = CMABAgent(agent_config, env.observation_space, env.action_space, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    load_path = config.get("benchmark", {}).get("load_checkpoint")
    if load_path:
        model_file = os.path.join(load_path, "models", f"{agent_type.lower()}_seed{seed}.pth")
        if os.path.exists(model_file):
            print(f"    [LOAD]: Resuming agent weights from checkpoint: {model_file}")
            agent.load(model_file)
        else:
            model_file_fallback = os.path.join(load_path, "models", f"{agent_type}_seed{seed}.pth")
            if os.path.exists(model_file_fallback):
                print(f"    [LOAD]: Resuming agent weights from checkpoint: {model_file_fallback}")
                agent.load(model_file_fallback)
            else:
                print(f"    [WARNING]: Checkpoint folder specified, but model file {model_file} not found! Starting from scratch.")

    buffer = None
    if agent_type.lower() in ["dqn", "qrdqn"]:
        buffer = ReplayBuffer(
            agent_config['buffer_size'], env.observation_space.shape,
            action_dim=env.action_space.n,
        )

    logger: CSVLogger | None = None
    if save_results:
        log_file = os.path.join(
            experiment_dir, "train",
            f"{agent_type}_{env_name.replace('/', '_')}_seed{seed}.csv",
        )
        headers = [
            "episode", "reward", "loss", "steps", "wall_time",
            "capacity", "rlf_rate", "ho_rate", "pp_rate",
            "reliability", "prep_rate", "res_reservation", "hof_rate",
        ]
        logger = CSVLogger(log_file, headers)

    num_episodes = agent_config.get('num_episodes', 20)
    eps_start = agent_config.get('epsilon_start', 1.0)
    eps_end = agent_config.get('epsilon_end', 0.05)
    eps_mult = agent_config.get('epsilon_mult', 0.99)

    # Calculate starting epsilon dynamically based on completed episodes of the loaded model
    completed_episodes = 0
    if load_path:
        try:
            import yaml
            with open(os.path.join(load_path, "config.yaml"), "r") as f:
                cfg = yaml.safe_load(f)
            completed_episodes = cfg.get("agent", {}).get("num_episodes", 2000)
        except Exception:
            completed_episodes = 2000

    epsilon = max(eps_end, eps_start * (eps_mult ** completed_episodes))
    batch_size = agent_config.get('batch_size', 64)
    train_freq = max(1, int(agent_config.get('train_freq', 4)))
    n_step = max(1, int(agent_config.get('n_step', 1)))
    gamma = float(agent_config.get('gamma', 0.99))

    start_time = time.time()
    rewards_history: list[float] = []
    global_step = 0

    try:
        for ep in range(num_episodes):
            state, _ = env.reset(seed=seed)
            agent.reset()
            if hasattr(agent, "set_env"):
                agent.set_env(env)
            episode_reward = 0
            episode_loss: list[Any] = []
            done = False
            last_info: dict[str, Any] = {}
            nstep_q: deque | None = deque() if n_step > 1 else None

            is_baseline = agent_type.lower() in ["ltm_baseline", "cmab"]
            while not done:
                if is_baseline:
                    next_state, reward, done, _, info = env.step(
                        0, high_res_callback=agent.select_action,
                    )
                else:
                    mask = env.valid_action_mask()
                    action = agent.select_action(state, epsilon, mask)
                    next_state, reward, done, _, info = env.step(action)
                    if buffer is not None:
                        next_mask = env.valid_action_mask()
                        if nstep_q is None:
                            buffer.push(state, action, reward, next_state, done, next_mask)
                        else:
                            nstep_q.append((state, action, reward, next_state, done))
                            if len(nstep_q) >= n_step:
                                _emit_nstep(buffer, nstep_q, gamma)
                            if done:
                                while nstep_q:
                                    _emit_nstep(buffer, nstep_q, gamma)
                    if (buffer is not None and len(buffer) > batch_size
                            and global_step % train_freq == 0):
                        metrics = agent.train_step(buffer.sample(batch_size, device=device))
                        episode_loss.append(metrics['loss'])
                state = next_state
                episode_reward += reward
                global_step += 1
                if done:
                    last_info = info

            metrics_8 = calculate_8_metrics(
                mcs_history=last_info["metrics"]["mcs"],
                rlf_history=last_info["metrics"]["rlf"],
                ho_history=last_info["metrics"]["ho"],
                hof_history=last_info["metrics"]["hof"],
                pp_history=last_info["metrics"]["pp"],
                reserved_history=last_info["metrics"]["reserved"],
                config=config,
            )

            epsilon = max(eps_end, epsilon * eps_mult)
            if episode_loss:
                avg_loss = float(torch.stack(episode_loss).mean().item())
            else:
                avg_loss = 0.0
            if logger is not None:
                logger.log([
                    ep + 1, episode_reward, avg_loss, env.t, time.time() - start_time,
                    metrics_8["capacity_avg"], metrics_8["rlf_rate"],
                    metrics_8["ho_rate"], metrics_8["pp_rate"],
                    metrics_8["reliability_pct"], metrics_8["prep_rate"],
                    metrics_8["res_reservation_pct"], metrics_8["hof_rate"],
                ])
            rewards_history.append(episode_reward)

            # Intermediate checkpoint and evaluation
            checkpoint_interval = config.get("benchmark", {}).get("checkpoint_interval", 2000)
            if (ep + 1) % checkpoint_interval == 0 and (ep + 1) < num_episodes:
                if save_results:
                    model_path = os.path.join(experiment_dir, "models", f"{agent_type}_seed{seed}_ep{ep + 1}.pth")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    agent.save(model_path)
                    
                    eval_prefix = os.path.join(experiment_dir, "eval", f"{agent_type}_seed{seed}_ep{ep + 1}")
                    run_evaluation(
                        agent, config, experiment_dir, agent_type, seed,
                        save_results=save_results, output_prefix=eval_prefix,
                        num_workers=eval_workers,
                    )

            pbar.update(1)
            pbar.set_postfix({
                "agent": agent_type,
                "seed": seed,
                "reward": f"{np.mean(rewards_history[-10:]):.1f}",
            })
    finally:
        if logger is not None:
            logger.close()

    if save_results:
        model_path = os.path.join(experiment_dir, "models", f"{agent_type}_seed{seed}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save(model_path)

    run_evaluation(
        agent, config, experiment_dir, agent_type, seed,
        save_results=save_results, num_workers=eval_workers,
    )

    env.close()
    return float(np.mean(rewards_history[-10:]))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _allocate_experiment_dir(results_dir: str, description: str) -> str:
    """Find the next free `bmk_YYYY-MM-DD_<num>_<desc>/` and return it."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = description.replace(" ", "-")
    num = 1
    while glob.glob(os.path.join(results_dir, f"bmk_{date_str}_{num}_*")):
        num += 1
    return os.path.join(results_dir, f"bmk_{date_str}_{num}_{slug}")


def _prepare_expdir(expdir: str, src_config_path: str) -> None:
    """Pre-create subdirectories and copy the config for provenance."""
    for sub in ("train", "eval", "models", "figures", "logs"):
        os.makedirs(os.path.join(expdir, sub), exist_ok=True)
    cfg_dst = os.path.join(expdir, "config.yaml")
    if not os.path.exists(cfg_dst):
        shutil.copy(src_config_path, cfg_dst)


def _default_eval_workers(num_parallel: int) -> int:
    """Pick a sane eval-worker count given K processes share a 12-core box.

    Each main.py process plus its workers must fit; we reserve 1 core per
    main process and split the remainder.
    """
    cpu_count = os.cpu_count() or 12
    return max(1, (cpu_count - num_parallel) // num_parallel)


def _spawn_worker(
    config_path: str, expdir: str, seed: int, agents: str,
    device: str, eval_workers: int, log_path: str,
    episodes: int | None = None,
    checkpoint_interval: int = 2000,
    load_checkpoint: str | None = None,
) -> subprocess.Popen:
    """Launch one worker subprocess that trains a single seed."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--config", config_path,
        "--experiment-dir", expdir,
        "--worker-seed", str(seed),
        "--agents", agents,
        "--device", device,
        "--eval-workers", str(eval_workers),
        "--checkpoint-interval", str(checkpoint_interval),
    ]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    if load_checkpoint is not None:
        cmd += ["--load-checkpoint", load_checkpoint]
    fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # own process group so we can kill cleanly
    )
    # Stash the log filehandle on the proc so the caller can close it.
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def _wait_with_signal_handling(procs: list[subprocess.Popen]) -> list[int]:
    """Wait for all subprocesses; on SIGINT/SIGTERM, kill them and exit."""
    def _terminate_all(*_: Any) -> None:
        print("\n[orchestrator] received signal, killing subprocesses...", flush=True)
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    codes = []
    for p in procs:
        codes.append(p.wait())
        fh = getattr(p, "_log_fh", None)
        if fh is not None:
            fh.close()
    return codes


def _run_inline(
    config: dict, expdir: str, agent_types: list[str], seeds: list[int],
    device: str, eval_workers: int | None, save_results: bool,
) -> None:
    """Run all (agent, seed) combinations in this process with a single
    tqdm bar. Used for single-seed or `--no-parallel` runs."""
    bench_cfg = config["benchmark"]
    num_episodes = config["agent"].get("num_episodes", 20)
    total_eps = len(agent_types) * len(seeds) * num_episodes
    with tqdm(total=total_eps, desc="Benchmark Overall") as pbar:
        for agent_type in agent_types:
            for seed in seeds:
                run_seed(
                    agent_type, bench_cfg["env_type"], seed, config, expdir,
                    pbar, device=device, save_results=save_results,
                    eval_workers=eval_workers,
                )


def _run_parallel(
    config_path: str, expdir: str, agents: str, seeds: list[int],
    device: str, eval_workers: int | None, episodes: int | None = None,
    checkpoint_interval: int = 2000,
    load_checkpoint: str | None = None,
) -> None:
    """Spawn one worker subprocess per seed; wait; aggregate."""
    if eval_workers is None:
        eval_workers = _default_eval_workers(len(seeds))
    print(f"[orchestrator] launching {len(seeds)} workers "
          f"(eval_workers={eval_workers} each)")
    procs = []
    for seed in seeds:
        log_path = os.path.join(expdir, "logs", f"seed{seed}.log")
        proc = _spawn_worker(
            config_path, expdir, seed, agents, device, eval_workers, log_path,
            episodes=episodes, checkpoint_interval=checkpoint_interval,
            load_checkpoint=load_checkpoint,
        )
        procs.append(proc)
        print(f"  -> seed {seed} (pid {proc.pid}) -> {log_path}")

    codes = _wait_with_signal_handling(procs)
    for seed, code in zip(seeds, codes):
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"[orchestrator] seed {seed}: {status}")
    if any(c != 0 for c in codes):
        print("[orchestrator] some workers failed; aggregator may produce "
              "incomplete plots.")


def run_benchmark() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--no_save", action="store_true",
                        help="Profile mode: don't write logs / models / plots")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--description", type=str, default="benchmark",
                        help="Short slug used in the bmk_ directory name")
    parser.add_argument("--agents", type=str, default="dqn,qrdqn",
                        help="Comma-separated agent types")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (e.g. '42,43,44'). "
                             "Defaults to benchmark.num_seeds in the config.")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Run seeds sequentially in this process instead "
                             "of forking one subprocess per seed.")
    parser.add_argument("--eval-workers", type=int, default=None,
                        help="Override eval worker count (default: "
                             "auto-picked based on parallelism).")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Reuse an existing experiment directory instead "
                             "of allocating a new one. Required by worker "
                             "subprocesses; rarely needed by users.")
    # Hidden flag: marks this process as a worker subprocess.
    parser.add_argument("--worker-seed", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override the number of training episodes")
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                        help="Save checkpoints and run greedy evaluation every K episodes")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to an existing benchmark directory to load weights from")
    args = parser.parse_args()

    Config.set_config_path(args.config)
    config = Config.get()

    # Dynamic path overrides for training on the 2000 UEs dataset
    train_dir = "data/ChannelGains_v2"
    for candidate in ["data/train", "data/5g_advanced_ltm_channel_trajectories/train", "data/ChannelGains_v2"]:
        if os.path.exists(candidate):
            train_dir = candidate
            break
    config['paths']['channel_data_directory'] = train_dir
    config['paths']['precomputed_data_directory'] = "data/Precomputed_v2"
    config['simulation']['ue_number'] = 2000
    config['benchmark']['checkpoint_interval'] = args.checkpoint_interval
    config['benchmark']['load_checkpoint'] = args.load_checkpoint
    
    if args.episodes is not None:
        config['agent']['num_episodes'] = args.episodes
    else:
        config['agent']['num_episodes'] = 2000
        
    config['benchmark']['num_seeds'] = 5

    # Cap intra-op threads to prevent thread contention when running multiple seeds in parallel.
    if args.device == "cpu":
        parallel_run = (args.worker_seed is not None) or (args.seeds is not None and len(args.seeds.split(',')) > 1) or (args.seeds is None and config["benchmark"]["num_seeds"] > 1)
        threads = 2 if parallel_run else 4
        torch.set_num_threads(threads)
    else:
        torch.set_num_threads(2)

    # -----------------------------------------------------------------
    # Worker mode: train exactly one seed in the given expdir, then exit.
    # -----------------------------------------------------------------
    if args.worker_seed is not None:
        if not args.experiment_dir:
            raise SystemExit("--worker-seed requires --experiment-dir")
        agent_types = [a.strip().lower() for a in args.agents.split(",")]
        num_episodes = config["agent"].get("num_episodes", 20)
        total_eps = len(agent_types) * num_episodes
        with tqdm(total=total_eps, desc=f"seed {args.worker_seed}") as pbar:
            for agent_type in agent_types:
                run_seed(
                    agent_type, config["benchmark"]["env_type"],
                    args.worker_seed, config, args.experiment_dir,
                    pbar, device=args.device, save_results=not args.no_save,
                    eval_workers=args.eval_workers,
                )
        return

    # -----------------------------------------------------------------
    # Orchestrator mode.
    # -----------------------------------------------------------------
    agent_types = [a.strip().lower() for a in args.agents.split(",")]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [42 + s for s in range(config["benchmark"]["num_seeds"])]

    if args.experiment_dir:
        expdir = args.experiment_dir
        _prepare_expdir(expdir, args.config)
    else:
        expdir = _allocate_experiment_dir(config["benchmark"]["results_dir"],
                                          args.description)
        if not args.no_save:
            _prepare_expdir(expdir, args.config)

    if not args.no_save:
        print("=== Starting Benchmark ===")
        print(f"  Directory: {expdir}")
        print(f"  Agents:    {args.agents}")
        print(f"  Device:    {args.device}")
        print(f"  Episodes:  {config['agent'].get('num_episodes', 1)}")
        print(f"  Seeds:     {seeds}")
        print(f"  Config:    {args.config}")
        parallel = (len(seeds) > 1) and not args.no_parallel
        print(f"  Mode:      {'parallel (subprocess per seed)' if parallel else 'sequential (inline)'}")
        print("==========================")
    else:
        print("=== Profiling Run (no artifacts) ===")

    bench_start = time.time()

    run_parallel = (len(seeds) > 1) and not args.no_parallel and not args.no_save
    if run_parallel:
        _run_parallel(
            args.config, expdir, args.agents, seeds, args.device,
            args.eval_workers, episodes=config["agent"].get("num_episodes"),
            checkpoint_interval=config["benchmark"].get("checkpoint_interval", 2000),
            load_checkpoint=config["benchmark"].get("load_checkpoint"),
        )
    else:
        _run_inline(
            config, expdir, agent_types, seeds, args.device,
            args.eval_workers, save_results=not args.no_save,
        )

    if not args.no_save:
        elapsed = time.time() - bench_start
        aggregate(
            expdir, config,
            extra_metadata={
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "total_execution_time_seconds": elapsed,
                "device": args.device,
                "agents": agent_types,
                "seeds": seeds,
                "parallel_mode": run_parallel,
            },
        )
        print(f"\nBenchmark completed in {elapsed:.0f}s. Artifacts: {expdir}")
    else:
        print("\nProfiling run completed. No artifacts saved.")


if __name__ == "__main__":
    run_benchmark()
