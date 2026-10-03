import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import re
import glob
from typing import Optional, Any


_AGENT_FROM_FILENAME_RE = re.compile(r'^(?P<agent>.+)_[^_]+_seed\d+\.csv$')


def _parse_agent_from_filename(filename: str) -> str:
    """Extract the agent type from a training-CSV filename.

    Training CSVs are named `<agent>_<env>_seed<N>.csv`, where the agent
    name itself may contain underscores (e.g. `ltm_baseline`). A naive
    `split('_')[0]` therefore mislabels multi-token agents. We strip the
    trailing `_<env>_seed<N>.csv` and keep the rest.
    """
    base = os.path.basename(filename)
    match = _AGENT_FROM_FILENAME_RE.match(base)
    if match:
        return match.group('agent')
    # Fallback for non-standard filenames
    return base.split('_seed')[0].rsplit('_', 1)[0]

def plot_learning_curves(results_dir: str, save_path: Optional[str] = None):
    """
    Parses all CSV files in results_dir, aggregates by agent, 
    and plots Reward vs Episode with confidence intervals and a 50-episode moving average.
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        print(f"No CSV files found in {results_dir}")
        return

    data_frames = []
    for f in files:
        df = pd.read_csv(f)
        df['agent'] = _parse_agent_from_filename(f)
        data_frames.append(df)

    all_data = pd.concat(data_frames, ignore_index=True)
    agents = all_data['agent'].unique()

    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        agent_data = all_data[all_data['agent'] == agent]
        grouped = agent_data.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        
        # Calculate Moving Average (Centered)
        window = 51
        grouped['moving_avg'] = grouped['mean'].rolling(window=window, min_periods=1, center=True).mean()
        
        # Plot Raw Mean (Transparent)
        line = plt.plot(grouped['episode'], grouped['mean'], alpha=0.3)[0]
        color = line.get_color()
        
        # Plot Moving Average (Solid)
        plt.plot(grouped['episode'], grouped['moving_avg'], color=color, label=f'{agent.upper()} (MA {window})', linewidth=2)
        
        # Plot Confidence Interval
        plt.fill_between(
            grouped['episode'], 
            grouped['mean'] - grouped['std'], 
            np.maximum(0, grouped['mean'] + grouped['std']),  # Ensure upper bound is not negative
            color=color,
            alpha=0.1
        )

    plt.title("Learning Curves: Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_efficiency(results_dir: str, metric: str = "reward", save_path: Optional[str] = None):
    """
    Plots a metric (reward or loss) against wall-clock time with a 50-unit moving average.
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    all_data = pd.concat([
        pd.read_csv(f).assign(agent=_parse_agent_from_filename(f)) for f in files
    ])
    agents = all_data['agent'].unique()

    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        agent_data = all_data[all_data['agent'] == agent]
        
        # Individual seeds (Very transparent)
        for seed_file in glob.glob(os.path.join(results_dir, f"{agent}_*.csv")):
            df = pd.read_csv(seed_file)
            plt.plot(df['wall_time'], df[metric], alpha=0.1, color='gray' if agent == 'dqn' else 'blue')
        
        # Aggregate trend
        grouped = agent_data.groupby('episode')[['wall_time', metric]].mean().reset_index()
        
        # Calculate Moving Average (Centered)
        window = 51
        grouped['moving_avg'] = grouped[metric].rolling(window=window, min_periods=1, center=True).mean()
        
        # Plot Aggregate Trend (Transparent)
        line = plt.plot(grouped['wall_time'], grouped[metric], alpha=0.3)[0]
        color = line.get_color()
        
        # Plot Moving Average (Solid)
        plt.plot(grouped['wall_time'], grouped['moving_avg'], color=color, label=f'{agent.upper()} (MA {window})', linewidth=2)

    plt.title(f"Efficiency: {metric.capitalize()} vs Wall-clock Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_metrics_grid(
    results_dir: str,
    save_path: Optional[str] = None,
    window: int = 51,
) -> None:
    """3x3 grid of per-episode learning curves across all 8 scientific
    metrics plus the training reward, aggregated across seeds per agent.

    Layout (left-to-right, top-to-bottom):
        reward, capacity, reliability,
        ho_rate, pp_rate, hof_rate,
        rlf_rate, prep_rate, res_reservation
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        print(f"No CSV files found in {results_dir}")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df['agent'] = _parse_agent_from_filename(f)
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)
    agents = all_data['agent'].unique()

    panels = [
        ("reward",          "Episode reward",        "reward (sum of per-step rewards)"),
        ("capacity",        "Capacity",              "bps/Hz"),
        ("reliability",     "Reliability",           "%"),
        ("ho_rate",         "HO rate",               "events/min"),
        ("pp_rate",         "Ping-pong rate",        "events/min"),
        ("hof_rate",        "HOF rate",              "events/min"),
        ("rlf_rate",        "RLF rate",              "events/min"),
        ("prep_rate",       "Cell preparation rate", "preps/min"),
        ("res_reservation", "Resource reservation",  "%"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    fig.suptitle("Training metrics by episode", fontsize=14, fontweight='bold')

    for ax, (col, title, ylabel) in zip(axes.flat, panels):
        if col not in all_data.columns:
            ax.set_title(f"{title} (missing)")
            ax.set_axis_off()
            continue
        for agent in agents:
            sub = all_data[all_data['agent'] == agent]
            grouped = sub.groupby('episode')[col].agg(['mean', 'std']).reset_index()
            grouped['ma'] = grouped['mean'].rolling(
                window=window, min_periods=1, center=True
            ).mean()

            raw_line, = ax.plot(grouped['episode'], grouped['mean'], alpha=0.2)
            color = raw_line.get_color()
            ax.plot(
                grouped['episode'], grouped['ma'],
                color=color, linewidth=2.0,
                label=f"{agent.upper()} (MA {window})",
            )
            if grouped['std'].notna().any():
                ax.fill_between(
                    grouped['episode'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=color, alpha=0.1,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("Episode")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='lower center',
            ncol=max(1, len(handles)), bbox_to_anchor=(0.5, -0.01),
            fontsize='medium',
        )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Metrics grid saved to {save_path}")
    else:
        plt.show()


def plot_quantiles(
    agent: Any,
    state: np.ndarray,
    action_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    top_k: int = 4,
):
    """Inverse-CDF view of each action's return distribution.

    Plots F^{-1}(tau) (predicted quantile values) against tau for the
    top-`top_k` actions by aggregated Q-value, plus annotations:
      - horizontal line at the aggregated Q-value (mean or CVaR
        depending on `agent.risk_type`).
      - for CVaR policies, a shaded band marking [0, risk_fraction]
        and a vertical line at tau = risk_fraction.
      - the chosen (argmax) action highlighted.

    Works for any quantile scheme (midpoint, gauss_legendre, trapezoidal,
    truncated CVaR) by reading `agent.scheme` for tau positions and the
    expectation / cvar weights.
    """
    state_tensor = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        predicted = agent.q_net(state_tensor)  # [1, A, num_predicted]
        scheme = agent.scheme
        # Full distribution including any fixed endpoints (trapezoidal).
        full = scheme.assemble_full(predicted)[0].cpu().numpy()  # [A, num_total]
        if agent.risk_type == "cvar":
            q_values = scheme.cvar(predicted)[0].cpu().numpy()
            agg_label = f"CVaR({agent.risk_fraction:.2g})"
        else:
            q_values = scheme.expectation(predicted)[0].cpu().numpy()
            agg_label = "Mean"

    # tau positions for the full assembled distribution.
    if scheme.has_fixed_endpoints:
        n = scheme.mean_weights.numel()
        full_tau = np.linspace(0.0, 1.0, n)
    else:
        full_tau = scheme.tau.cpu().numpy()

    num_actions = full.shape[0]
    chosen = int(np.argmax(q_values))
    order = np.argsort(-q_values)
    top = order[: min(top_k, num_actions)].tolist()
    if chosen not in top:
        top[-1] = chosen  # always show the picked action

    plt.figure(figsize=(12, 6))
    palette = plt.cm.tab10.colors
    for rank, i in enumerate(top):
        color = palette[rank % len(palette)]
        label = action_names[i] if action_names else f"Action {i}"
        label = f"{label}  ({agg_label}={q_values[i]:.3f})"
        if i == chosen:
            label = "* " + label
        plt.plot(full_tau, full[i], marker="o", ms=3, color=color, label=label)
        plt.axhline(q_values[i], color=color, alpha=0.25, linestyle="--")

    if agent.risk_type == "cvar":
        plt.axvspan(0.0, agent.risk_fraction, color="red", alpha=0.08,
                    label=f"CVaR mass < {agent.risk_fraction}")
        plt.axvline(agent.risk_fraction, color="red", alpha=0.4,
                    linestyle=":")

    plt.title(
        f"Return-distribution quantiles  (mode={scheme.mode}, "
        f"chosen=Action {chosen}, num_predicted={scheme.num_predicted})"
    )
    plt.xlabel(r"Quantile fraction $\tau$")
    plt.ylabel(r"Predicted $F^{-1}(\tau)$ (return)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
        print(f"Quantile plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="Directory with benchmark CSVs")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "quantiles"])
    args = parser.parse_args()

    results_root = "results/benchmarks"
    
    if args.mode == "benchmark":
        target_dir = args.results_dir
        if not target_dir:
            subdirs = [os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
            if not subdirs:
                print("No benchmark results found.")
                exit()
            target_dir = max(subdirs, key=os.path.getmtime)
            
        print(f"Generating benchmark plots for: {target_dir}")
        plot_learning_curves(target_dir, save_path=os.path.join(target_dir, "learning_curves.png"))
        plot_efficiency(target_dir, metric="reward", save_path=os.path.join(target_dir, "reward_vs_time.png"))
        plot_efficiency(target_dir, metric="loss", save_path=os.path.join(target_dir, "loss_vs_time.png"))
        plot_metrics_grid(target_dir, save_path=os.path.join(target_dir, "metrics_grid.png"))
    
    elif args.mode == "quantiles":
        print("Quantile plotting requires an active agent instance.")
