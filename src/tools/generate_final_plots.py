import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.patches import Patch

def load_summary(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index('metric')
    df.index = [i.strip() for i in df.index]
    
    series = df['mean']
    # If the reward is the total episode reward (e.g. > 50), convert it to reward per decision
    if 'reward' in series.index and 'n_decisions' in series.index:
        if series['reward'] > 50.0:
            series = series.copy()
            series['reward'] = series['reward'] / max(series['n_decisions'], 1.0)
    elif 'reward' in series.index and series['reward'] > 50.0:
        # Fallback if n_decisions is not present (standard 300 decisions)
        series = series.copy()
        series['reward'] = series['reward'] / 300.0
        
    return series


# Canonical KPI order, used across every table and plot in the paper:
# benefits (higher is better) -> mobility outcomes by increasing severity
# -> resource cost.
METRICS_TO_PLOT = [
    "capacity_avg", "reliability_pct", "hof_rate",
    "rlf_rate", "ho_rate", "pp_rate",
    "prep_rate", "res_reservation_pct", "reward",
]
NICE_NAMES = {
    "reward": "Reward / Decision",
    "capacity_avg": "Capacity (bps/Hz)",
    "reliability_pct": "Reliability (%)",
    "ho_rate": "HO Rate (/min)",
    "pp_rate": "PP Rate (/min)",
    "hof_rate": "HOF Rate (/min)",
    "rlf_rate": "RLF Rate (/min)",
    "prep_rate": "Preparations (/min)",
    "res_reservation_pct": "Res. Reservation (%)",
}


def _load_data(mapping):
    """Load every summary CSV whose file-key is in `mapping`, keyed by label.

    Returns a DataFrame indexed by the nice label, columns = metrics.
    """
    results_dir = "results/final_metrics"
    all_data = {}
    for f in glob.glob(os.path.join(results_dir, "*.csv")):
        key = os.path.basename(f).replace(".csv", "")
        # Handle specific naming like dqn_2k-ep_summary.
        if "qrdqn_2k-ep" in key:
            key = "qrdqn_summary"
        elif "dqn_2k-ep" in key:
            key = "dqn_summary"
        if key in mapping:
            series = load_summary(f)
            if series is not None:
                all_data[mapping[key]] = series
    return pd.DataFrame(all_data).transpose()


CUSTOM_YLIMS = {
    "reward": (3.0, None),
    "reliability_pct": (90.0, 100.0),
    "capacity_avg": (2.0, None),
}


def add_broken_axis_mark(ax, is_small=False):
    """Draw two parallel slanted tick marks (//) across the y-axis spine at the bottom."""
    dx = 0.020 if is_small else 0.012
    dy = 0.016 if is_small else 0.009
    h = 0.024 if is_small else 0.015
    y0 = 0.015
    ax.plot((-dx, +dx), (y0 - dy, y0 + dy), transform=ax.transAxes,
            color='black', clip_on=False, lw=0.8 if is_small else 1.2)
    ax.plot((-dx, +dx), (y0 + h - dy, y0 + h + dy), transform=ax.transAxes,
            color='black', clip_on=False, lw=0.8 if is_small else 1.2)


def _draw_bar_grid(data, agents, agent_color, agent_hatch, title, out_name,
                   legend_handles, bar_width=0.8, metrics=None,
                   grid=(3, 3), figsize=(22, 18)):
    """Grid of KPI bar charts for the given agent order.

    Custom y-limits and broken-axis marks are applied to zoomed metrics
    (reward, reliability); all other panels start at zero. Missing values leave
    a reserved-but-empty slot.

    Args:
        metrics: KPI keys to plot (defaults to all nine, in canonical order).
        grid: (nrows, ncols) of the subplot layout.
        figsize: overall figure size.
    """
    if metrics is None:
        metrics = METRICS_TO_PLOT
    fig, axes = plt.subplots(grid[0], grid[1], figsize=figsize)
    axes = axes.flatten()
    colors = [agent_color[a] for a in agents]
    hatches = [agent_hatch[a] for a in agents]
    x_pos = np.arange(len(agents))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric not in data.columns:
            ax.set_visible(False)
            continue
        series = data[metric].reindex(agents)
        bars = ax.bar(x_pos, series.values, width=bar_width,
                      color=colors, alpha=0.8, edgecolor='black')
        # Per-bar hatch (older matplotlib rejects a list for `hatch=`).
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents)
        ax.set_xlim(-0.6, len(agents) - 0.4)
        valid_vals = [v for v in series.values if v is not None and not pd.isna(v)]
        max_v = max(valid_vals) if len(valid_vals) > 0 else 1.0
        if metric in CUSTOM_YLIMS:
            ymin, ymax = CUSTOM_YLIMS[metric]
            if ymax is None:
                ymax = max_v + (max_v - ymin) * 0.15
            ax.set_ylim(ymin, ymax)
            add_broken_axis_mark(ax, is_small=False)
        else:
            min_v = min(valid_vals) if len(valid_vals) > 0 else 0.0
            ax.set_ylim(min(0.0, min_v), max_v * 1.15)
        ax.set_title(NICE_NAMES[metric], fontsize=15, fontweight='bold', pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        for bar, yval in zip(bars, series.values):
            if yval is None or pd.isna(yval):
                continue
            fmt = '.3f' if yval < 1 else '.2f'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + y_range * 0.015,
                f'{yval:{fmt}}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

    # Hide any leftover panels (when metrics < grid cells).
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.legend(
        handles=legend_handles, loc='lower center', ncol=len(legend_handles),
        fontsize=13.5, frameon=True, bbox_to_anchor=(0.5, 0.010),
    )
    plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plots_dir = "results/final_metrics/plots"
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

def generate_plots():
    results_dir = "results/final_metrics"
    
    mapping = {
        "ltm_ours_summary": "LTM",
        "cmab_ours_summary": "CMAB",
        "dqn_summary": "DQN",
        "qrdqn_riskneutral_summary": "RN",
        "qrdqn_trunc_a025_summary": "Truncated (k=7)",
        "qrdqn_cvar_a025_summary": "Hard CVaR",
        "qrdqn_softstep_summary": "Soft-step",
    }

    desired_order = [
        "LTM", "CMAB", "DQN", "RN", "Hard CVaR", "Truncated (k=7)", "Soft-step"
    ]

    agent_color = {
        "LTM": "#7f7f7f",             # Neutral Grey (Baseline)
        "CMAB": "#1f77b4",            # Classic Blue (Contextual Bandit)
        "DQN": "#ff7f0e",             # Vibrant Orange
        "RN": "#8c564b",              # Warm Sienna Brown (Risk-Neutral)
        "Hard CVaR": "#e377c2",       # Soft Pink / Magenta
        "Truncated (k=7)": "#9467bd", # Royal Purple (Risk-Aware Truncated)
        "Soft-step": "#d62728",       # Crimson Red (Proposed Champion)
    }
    agent_hatch = {a: "" for a in desired_order}
    
    data = _load_data(mapping).reindex(desired_order)
    agents = desired_order
    colors = [agent_color[a] for a in agents]

    legend_handles = [
        Patch(facecolor="#7f7f7f", edgecolor='black', alpha=0.85, label="3GPP (LTM)"),
        Patch(facecolor="#1f77b4", edgecolor='black', alpha=0.85, label="CMAB (Bandit)"),
        Patch(facecolor="#ff7f0e", edgecolor='black', alpha=0.85, label="Scalar DQN"),
        Patch(facecolor="#8c564b", edgecolor='black', alpha=0.85, label="RN QR-DQN"),
        Patch(facecolor="#e377c2", edgecolor='black', alpha=0.85, label="Hard CVaR"),
        Patch(facecolor="#9467bd", edgecolor='black', alpha=0.85, label="Truncated (k=7)"),
        Patch(facecolor="#d62728", edgecolor='black', alpha=0.85, label="Soft-step (Proposed)"),
    ]
    print(f"Generating 3x3 bar plots for agents in order: {agents}")
    _draw_bar_grid(
        data, agents, agent_color, agent_hatch,
        "LTM-HO Comparative Analysis: RL Agents vs. Baselines",
        "master_bar_plots.png", legend_handles,
    )

    # --- 3. CONSOLIDATED RADAR CHART ---
    plots_dir = "results/final_metrics/plots"
    print("Generating master radial chart...")
    radar_labels = ["Capacity", "Reliability", "HO Stability", "PP Stability", "HOF (Inv)", "RLF (Inv)"]
    radar_df = pd.DataFrame(index=agents, columns=radar_labels)
    
    for agent in agents:
        radar_df.loc[agent, "Capacity"] = data.loc[agent, "capacity_avg"]
        radar_df.loc[agent, "Reliability"] = data.loc[agent, "reliability_pct"]
        radar_df.loc[agent, "RLF (Inv)"] = 1.0 / (1.0 + data.loc[agent, "rlf_rate"])
        radar_df.loc[agent, "HOF (Inv)"] = 1.0 / (1.0 + data.loc[agent, "hof_rate"])
        radar_df.loc[agent, "HO Stability"] = 1.0 / (1.0 + data.loc[agent, "ho_rate"] / 20.0)
        radar_df.loc[agent, "PP Stability"] = 1.0 / (1.0 + data.loc[agent, "pp_rate"])

    for col in radar_df.columns:
        c_min, c_max = radar_df[col].min(), radar_df[col].max()
        if c_max > c_min:
            radar_df[col] = (radar_df[col] - c_min) / (c_max - c_min) * 0.7 + 0.3
        else:
            radar_df[col] = 1.0

    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    for i, agent in enumerate(agents):
        values = radar_df.loc[agent].tolist()
        values += values[:1]
        lw = 4 if agent == "Soft-step" else 2.5
        alpha_fill = 0.18 if agent == "Soft-step" else 0.05
        ax.plot(angles, values, color=colors[i], linewidth=lw, label=agent, alpha=0.95)
        ax.fill(angles, values, color=colors[i], alpha=alpha_fill)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=13, fontweight='bold')
    
    plt.title("Performance Profile: Global Multi-Objective Comparison", fontsize=20, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11, frameon=True)
    
    radar_path = os.path.join(plots_dir, "master_radial_plot.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {radar_path}")

def generate_ltm_cmab():
    """2x4 KPI bar grid for LTM vs CMAB.

    Wide 2-row x 4-col layout (the 8 KPIs except reward).
    """
    mapping = {
        "ltm_ours_summary": "LTM",
        "cmab_ours_summary": "CMAB",
    }
    order = ["LTM", "CMAB"]
    agent_color = {
        "LTM": "#7f7f7f",
        "CMAB": "#1f77b4",
    }
    agent_hatch = {
        "LTM": "",
        "CMAB": "",
    }
    data = _load_data(mapping).reindex(order)

    legend_handles = [
        Patch(facecolor="#7f7f7f", edgecolor='black', alpha=0.85,
              label="3GPP Baseline (LTM)"),
        Patch(facecolor="#1f77b4", edgecolor='black', alpha=0.85,
              label="Contextual Bandit (CMAB)"),
    ]
    metrics = ["capacity_avg", "reliability_pct", "ho_rate", "pp_rate",
               "hof_rate", "rlf_rate", "prep_rate", "res_reservation_pct"]
    print(f"Generating LTM-vs-CMAB 2x4 bar plots: {order}")
    _draw_bar_grid(
        data, order, agent_color, agent_hatch,
        "Baselines Comparison: LTM vs CMAB",
        "master_bar_plots_ltm_cmab.png", legend_handles, bar_width=0.4,
        metrics=metrics, grid=(2, 4), figsize=(20, 10),
    )


def generate_distributional(include_rn1q=True,
                            out_name="master_bar_plots_distributional.png"):
    """3x3 KPI bar grid for the distribution ablation."""
    mapping = {
        "ltm_ours_summary": "LTM",
        "cmab_ours_summary": "CMAB",
        "dqn_summary": "DQN",
        "qrdqn_rn_1q_summary": "RN-1q",
        "qrdqn_riskneutral_summary": "RN",
    }
    order = ["LTM", "CMAB", "DQN", "RN-1q", "RN"]
    agent_color = {
        "LTM": "#7f7f7f",        # grey
        "CMAB": "#1f77b4",       # blue
        "DQN": "#ff7f0e",        # orange
        "RN-1q": "#bcbd22",      # olive / yellow-green
        "RN": "#8c564b",         # brown
    }
    legend_by_agent = {
        "LTM": Patch(facecolor="#7f7f7f", edgecolor='black', alpha=0.85,
                     label="3GPP Baseline (LTM)"),
        "CMAB": Patch(facecolor="#1f77b4", edgecolor='black', alpha=0.85,
                      label="Contextual Bandit (CMAB)"),
        "DQN": Patch(facecolor="#ff7f0e", edgecolor='black', alpha=0.85,
                     label="Scalar DQN"),
        "RN-1q": Patch(facecolor="#bcbd22", edgecolor='black', alpha=0.85,
                       label="RN-1q (Single Quantile)"),
        "RN": Patch(facecolor="#8c564b", edgecolor='black', alpha=0.85,
                    label="RN QR-DQN (N=25)"),
    }
    if not include_rn1q:
        order = [a for a in order if a != "RN-1q"]
        mapping = {k: v for k, v in mapping.items() if v != "RN-1q"}

    agent_hatch = {a: "" for a in order}
    data = _load_data(mapping).reindex(order)
    legend_handles = [legend_by_agent[a] for a in order]
    print(f"Generating 3x3 distributional bar plots: {order}")
    _draw_bar_grid(
        data, order, agent_color, agent_hatch,
        "Distributional Learning on Handover: DQN vs QR-DQN",
        out_name, legend_handles, bar_width=0.6,
    )


def generate_single_column_bars():
    """Column-width variant of the master bar plot (single IEEE text column)."""
    results_dir = "results/final_metrics"
    mapping = {
        "ltm_ours_summary": "LTM",
        "cmab_ours_summary": "CMAB",
        "dqn_summary": "DQN",
        "qrdqn_riskneutral_summary": "RN",
        "qrdqn_trunc_a025_summary": "Trunc (k=7)",
        "qrdqn_cvar_a025_summary": "Hard CVaR",
        "qrdqn_softstep_summary": "Soft-step",
    }
    order = ["LTM", "CMAB", "DQN", "RN", "Hard CVaR", "Trunc (k=7)", "Soft-step"]
    color = {
        "LTM": "#7f7f7f",
        "CMAB": "#1f77b4",
        "DQN": "#ff7f0e",
        "RN": "#8c564b",
        "Hard CVaR": "#e377c2",
        "Trunc (k=7)": "#9467bd",
        "Soft-step": "#d62728",
    }
    hatches = ["" for a in order]

    all_data = {}
    for f in glob.glob(os.path.join(results_dir, "*.csv")):
        key = os.path.basename(f).replace(".csv", "")
        if key in mapping:
            s = load_summary(f)
            if s is not None:
                all_data[mapping[key]] = s
    data = pd.DataFrame(all_data).transpose().reindex(order)

    metrics = ["capacity_avg", "reliability_pct",
               "hof_rate", "rlf_rate",
               "ho_rate", "pp_rate",
               "prep_rate", "res_reservation_pct",
               "reward"]
    nice = {
        "reward": "Reward / Decision", "capacity_avg": "Capacity (bps/Hz)",
        "reliability_pct": "Reliability (%)", "ho_rate": "HO (/min)",
        "pp_rate": "PP (/min)", "hof_rate": "HOF (/min)",
        "rlf_rate": "RLF (/min)", "prep_rate": "Preparations (/min)",
        "res_reservation_pct": "Res. Reservation (%)",
    }

    x = np.arange(len(order))
    colors = [color[a] for a in order]

    fig, axes = plt.subplots(5, 2, figsize=(3.5, 4.8))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = data[metric].reindex(order).values
        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5, width=0.82)
        ax.set_title(nice[metric], fontsize=6.8, fontweight="bold", pad=1.5)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=5.6, pad=1.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.4)
        ax.set_xlim(-0.55, len(order) - 0.45)

        valid_vals = [v for v in vals if v is not None and not pd.isna(v)]
        max_v = max(valid_vals) if len(valid_vals) > 0 else 1.0
        if metric in CUSTOM_YLIMS:
            ymin, ymax = CUSTOM_YLIMS[metric]
            if ymax is None:
                ymax = max_v + (max_v - ymin) * 0.15
            ax.set_ylim(ymin, ymax)
            add_broken_axis_mark(ax, is_small=True)
        else:
            min_v = min(valid_vals) if len(valid_vals) > 0 else 0.0
            ax.set_ylim(min(0.0, min_v), max_v * 1.15)

        # Clean y-axis tick formatting to prevent horizontal collisions
        if metric == "reliability_pct":
            ax.set_yticks([90, 95, 100])
            ax.set_yticklabels(["90", "95", "100"])
        elif metric == "rlf_rate":
            ax.set_yticks([0.0, 0.1])
            ax.set_yticklabels(["0.0", "0.1"])
        elif metric == "capacity_avg":
            ax.set_yticks([2, 3, 4])
            ax.set_yticklabels(["2", "3", "4"])
        elif metric == "reward":
            ax.set_yticks([3.0, 3.3, 3.6])
            ax.set_yticklabels(["3.0", "3.3", "3.6"])

    legend_handles = [
        Patch(facecolor="#7f7f7f", edgecolor="black", alpha=0.85, label="3GPP Baseline (LTM)"),
        Patch(facecolor="#1f77b4", edgecolor="black", alpha=0.85, label="Contextual Bandit (CMAB)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", alpha=0.85, label="Scalar DQN"),
        Patch(facecolor="#8c564b", edgecolor="black", alpha=0.85, label="RN QR-DQN (N=25)"),
        Patch(facecolor="#e377c2", edgecolor="black", alpha=0.85, label="Hard CVaR (alpha=0.25)"),
        Patch(facecolor="#9467bd", edgecolor="black", alpha=0.85, label="Truncated CVaR (k=7)"),
        Patch(facecolor="#d62728", edgecolor="black", alpha=0.85, label="Soft-step QR-DQN (Proposed)"),
    ]
    axes[9].axis("off")
    axes[9].legend(handles=legend_handles, loc="center", fontsize=5.6,
                   frameon=True, framealpha=0.95, edgecolor="gray",
                   handlelength=1.0, borderpad=0.3, labelspacing=0.25)

    fig.tight_layout(h_pad=0.25, w_pad=0.70)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, "master_bar_plots_1col.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    generate_plots()
    generate_ltm_cmab()
    generate_distributional()
    generate_distributional(include_rn1q=False,
                            out_name="master_bar_plots_distributional_no1q.png")
    generate_single_column_bars()
