#!/usr/bin/env python3
"""Generate the official Figure 2: Capacity vs. Handover Failure Rate (HOF).

Plots the true physical Pareto frontier of 5G-Advanced LTM mobility on 1,000
unseen test trajectories across 5 independent seeds with cross-seed error bars.
Strictly adheres to the canonical color palette and IEEE Transactions styling.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_pareto_frontier():
    # Canonical cross-seed evaluation data from Table II (5 seeds, 1000 unseen UEs)
    models = {
        '3GPP (LTM)': {
            'hof': 1.151, 'hof_std': 0.080, # Cross-seed std
            'cap': 3.753, 'cap_std': 0.015,
            'color': '#7f7f7f', 'marker': 's', 'size': 50, 'zorder': 4
        },
        'CMAB (Bandit)': {
            'hof': 0.775, 'hof_std': 0.035,
            'cap': 3.756, 'cap_std': 0.020,
            'color': '#1f77b4', 'marker': '^', 'size': 55, 'zorder': 4
        },
        'Scalar DQN': {
            'hof': 1.116, 'hof_std': 0.085,
            'cap': 4.031, 'cap_std': 0.006,
            'color': '#ff7f0e', 'marker': 'o', 'size': 55, 'zorder': 4
        },
        'RN QR-DQN': {
            'hof': 0.769, 'hof_std': 0.065,
            'cap': 4.044, 'cap_std': 0.005,
            'color': '#8c564b', 'marker': 'P', 'size': 60, 'zorder': 5
        },
        r'Hard CVaR ($\alpha = 0.25$)': {
            'hof': 0.790, 'hof_std': 0.055,
            'cap': 4.008, 'cap_std': 0.010,
            'color': '#e377c2', 'marker': 'D', 'size': 50, 'zorder': 4
        },
        'Truncated (k=7)': {
            'hof': 0.905, 'hof_std': 0.090,
            'cap': 4.000, 'cap_std': 0.006,
            'color': '#9467bd', 'marker': 'v', 'size': 55, 'zorder': 4
        },
        'Soft-step (Proposed)': {
            'hof': 0.744, 'hof_std': 0.040,
            'cap': 4.041, 'cap_std': 0.006,
            'color': '#d62728', 'marker': '*', 'size': 105, 'zorder': 6
        },
    }

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 10.5,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 8.8,
        'figure.titlesize': 12,
    })

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)

    for name, d in models.items():
        ew = 1.2 if 'Soft-step' in name else 0.8
        # Error bars for statistical cross-seed significance
        ax.errorbar(
            d['hof'], d['cap'], xerr=d['hof_std'], yerr=d['cap_std'],
            fmt='none', ecolor=d['color'], elinewidth=1.0, capsize=2.5,
            capthick=0.8, alpha=0.75, zorder=3
        )
        ax.scatter(
            d['hof'], d['cap'], color=d['color'], marker=d['marker'],
            s=d['size'], edgecolor='black', linewidth=ew, label=name,
            zorder=d['zorder']
        )

    ax.set_xlabel('Handover Failure Rate [HOF / min] (← Lower is Better)', fontweight='bold')
    ax.set_ylabel('Spectral Efficiency / Capacity [bps/Hz]\n(↑ Higher is Better)', fontweight='bold')
    ax.set_title('5G-Advanced LTM Pareto Frontier: Capacity vs. Failure Rate', fontweight='bold', pad=10)

    ax.set_xlim(0.70, 1.25)
    ax.set_ylim(3.70, 4.09)
    ax.grid(True, linestyle='--', alpha=0.45, zorder=0)

    # Elegant legend placement in the empty middle-right region
    ax.legend(
        loc='center right', bbox_to_anchor=(0.98, 0.42), frameon=True, framealpha=0.92,
        edgecolor='#cccccc', borderpad=0.5, labelspacing=0.35, markerscale=1.0
    )

    plt.tight_layout()

    out_dirs = [
        'paper_tmlcn',
        'results/final_metrics/plots',
        'docs/assets',
    ]

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, 'risk_frontier.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved {path}")

    plt.close()

if __name__ == '__main__':
    generate_pareto_frontier()
