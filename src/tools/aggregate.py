"""Post-training aggregator for parallel multi-seed benchmarks.

Walks an experiment directory and:
- Picks the best model per agent (by mean eval reward across UEs) and
  copies it to `models/{agent}_best.pth`.
- Renders the learning-curves / efficiency / metrics-grid plots
  (which already aggregate mean+std across seeds via plot.py).
- Renders the QR-DQN return-distribution plot if applicable.
- Writes `metadata.json` summarising the run.

Called inline by main.py at the end of every benchmark run. Can also be
invoked standalone to rebuild artifacts without re-training:

    python src/tools/aggregate.py --experiment-dir results/benchmarks/bmk_...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from typing import Any

import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.distrl.utils.config import Config
from src.distrl.viz.plot import (
    plot_efficiency,
    plot_learning_curves,
    plot_metrics_grid,
    plot_quantiles,
)


_TRAIN_CSV_RE = re.compile(r"^(?P<agent>.+)_[^_]+_seed(?P<seed>\d+)\.csv$")
_EVAL_SUMMARY_RE = re.compile(r"^(?P<agent>.+)_summary_seed(?P<seed>\d+)\.csv$")


def _discover_agents(train_dir: str) -> list[str]:
    """Return the sorted unique agent names found in `train_dir`."""
    agents: set[str] = set()
    for path in glob.glob(os.path.join(train_dir, "*.csv")):
        m = _TRAIN_CSV_RE.match(os.path.basename(path))
        if m:
            agents.add(m.group("agent"))
    return sorted(agents)


def _pick_best_model(expdir: str, agent: str) -> int | None:
    """Copy the seed with highest mean eval reward to `{agent}_best.pth`.

    Returns the chosen seed, or None if no eval summary / model was found.
    """
    candidates: list[tuple[int, float]] = []
    pattern = os.path.join(expdir, "eval", f"{agent}_summary_seed*.csv")
    for path in glob.glob(pattern):
        m = _EVAL_SUMMARY_RE.match(os.path.basename(path))
        if not m:
            continue
        seed = int(m.group("seed"))
        df = pd.read_csv(path)
        row = df[df["metric"] == "reward"]
        if row.empty:
            continue
        candidates.append((seed, float(row.iloc[0]["mean"])))

    if not candidates:
        return None
    best_seed, _ = max(candidates, key=lambda t: t[1])

    src = os.path.join(expdir, "models", f"{agent}_seed{best_seed}.pth")
    if not os.path.exists(src):
        return None
    dst = os.path.join(expdir, "models", f"{agent}_best.pth")
    shutil.copy(src, dst)
    return best_seed


def _render_quantile_plot(expdir: str, config: dict) -> None:
    """Render the QR-DQN inverse-CDF plot using the best model."""
    best_path = os.path.join(expdir, "models", "qrdqn_best.pth")
    if not os.path.exists(best_path):
        return
    # Local imports to avoid loading torch / env when not needed.
    from src.distrl.agents.distributional.qrdqn import QRDQNAgent
    from src.distrl.envs.ltm_gym import LTMEnv

    env = LTMEnv(config=config)
    agent = QRDQNAgent(config["agent"], env.observation_space, env.action_space)
    agent.load(best_path)

    state, _ = env.reset()
    for _ in range(50):
        action = agent.select_action(state, 0)
        state, _, done, _, _ = env.step(action)
        if done:
            break

    save_path = os.path.join(expdir, "figures", "quantile_distribution.png")
    plot_quantiles(agent, state, save_path=save_path)
    env.close()


def _write_metadata(expdir: str, config: dict, extras: dict[str, Any]) -> None:
    metadata = {"config": config, **extras}
    with open(os.path.join(expdir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=4, default=str)


def aggregate(
    expdir: str,
    config: dict,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Generate every post-training artifact in `expdir`.

    Idempotent: safe to call multiple times. Assumes train/, eval/, and
    models/ are already populated by worker runs.
    """
    train_dir = os.path.join(expdir, "train")
    fig_dir = os.path.join(expdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    agents = _discover_agents(train_dir)
    if not agents:
        print(f"aggregate: no training CSVs found in {train_dir}; skipping")
        return
    print(f"aggregate: agents found = {', '.join(agents)}")

    for agent in agents:
        chosen = _pick_best_model(expdir, agent)
        if chosen is not None:
            print(f"  -> {agent}_best.pth (from seed {chosen}, picked by eval reward)")
        else:
            print(f"  ! {agent}: no eval summary / model found, skipping best-model")

    print("aggregate: rendering plots")
    plot_learning_curves(
        train_dir, save_path=os.path.join(fig_dir, "learning_curves.png"),
    )
    plot_efficiency(
        train_dir, metric="reward",
        save_path=os.path.join(fig_dir, "reward_vs_time.png"),
    )
    plot_efficiency(
        train_dir, metric="loss",
        save_path=os.path.join(fig_dir, "loss_vs_time.png"),
    )
    plot_metrics_grid(
        train_dir, save_path=os.path.join(fig_dir, "metrics_grid.png"),
    )

    if "qrdqn" in agents:
        _render_quantile_plot(expdir, config)

    _write_metadata(expdir, config, extra_metadata or {})
    print(f"aggregate: done. artifacts in {expdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to a benchmark directory (must contain config.yaml).",
    )
    args = parser.parse_args()

    cfg_path = os.path.join(args.experiment_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"No config.yaml at {cfg_path}")

    Config.set_config_path(cfg_path)
    aggregate(args.experiment_dir, Config.get())


if __name__ == "__main__":
    main()
