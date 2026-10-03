# src/tools/

Curated utility tools and plot generators for the IEEE TMLCN paper and analysis.

Always export `PYTHONPATH` from the repository root before running any tool:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
./venv-RL/bin/python3 src/tools/<subfolder>/<script>.py
```

---

## Subdirectories

### 1. `data/` — Dataset Preprocessing Pipeline
* **`preprocess_dataset.py`**: One-time precomputation script. Converts raw `data/ChannelGains/*.mat` into fast `data/Precomputed/{no_ris,with_ris}/User*_precomputed.npz` archives with cryptographic `physics_hash()` validation.

### 2. `paper/` — Canonical Manuscript Figures & Tables
* **`generate_final_plots.py`**: Reads `results/final_metrics/*_summary.csv` and renders the canonical publication comparison figures into `results/final_metrics/plots/` and `paper_tmlcn/figures/`:
  * `master_bar_plots_1col.png` (the 1-column 3GPP KPI overview in Section VI.D)
  * `master_bar_plots.png` (full 3×3 metric grid)
  * `master_radial_plot.png` (tradeoff radar plot)
  * `master_bar_plots_distributional.png` (distributional comparison)
* **`generate_legacy_summary.py`**: Runs the 3GPP standardized heuristic baseline over all 1,000 evaluation UEs to write `results/final_metrics/legacy_baseline_summary.csv`.
* **`plot_risk_frontier.py`**: Generates Fig. 4 in the manuscript: the two-panel Risk Frontier plotting Reward vs. Handover Failure rate and tail outage.
* **`plot_softstep_tradeoff_analysis.py`**: Produces the 4-panel multi-axis tradeoff analysis for the proposed Soft-step QR-DQN policy across risk parameters.
* **`plot_softstep_sensitivity_grid.py`**: Evaluates parameter grid stability across risk fraction $\alpha \in \{0.10, 0.25, 0.50\}$ and spectral boost $r \in \{1.5, 2.0, 3.0\}$.
* **`plot_softstep_pareto.py`**: Generates the Pareto frontier isolating non-dominated risk/throughput configurations.

### 3. `analysis/` — Single-Trajectory Visualizers & Distributions
* **`plot_policy_comparison.py`**: Interactive head-to-head trajectory comparator across any two policies (e.g. LTM vs. Soft-step on UE 900) with animated MP4 generation support (`--animate`).
* **`gen_head_to_head_set.py`**: Batch generator for the standard presentation suite (4 policy pairs $\times$ 3 UEs).
* **`gen_single_policy_set.py`**: Generates standalone single-policy trajectory rasters.
* **`plot_return_density.py`**: Visualizes learned per-action return distribution probability density functions (PDFs).
* **`plot_return_distributions.py`**: Analyzes learned return spread vs. instantaneous channel and radio conditions.
* **`plot_per_ue_tails.py`**: Evaluates tail performance over the worst 100 user trajectories.
* **`plot_finals_learning_overlay.py`**: Overlays multi-seed training curves to evaluate convergence stability across seeds.
