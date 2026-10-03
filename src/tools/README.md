# src/tools/

Standalone utility scripts. None of these are imported by the main `src/distrl/`
package — they're command-line tools you run by hand. Always export
`PYTHONPATH` first (see `CLAUDE.md`):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 src/tools/<script>.py
```

## Data pipeline
- **`preprocess_dataset.py`** — One-time: convert raw `data/ChannelGains/*.mat`
  into `data/Precomputed/{no_ris,with_ris}/User*_precomputed.npz`. Stamps the
  current `physics_hash()` into each file; the env refuses to load a cache whose
  hash doesn't match the live constants.

## Final-comparison plots (master results)
- **`generate_final_plots.py`** — Read the `*_summary.csv` files in
  `results/final_metrics/` and render the canonical comparison figures into
  `results/final_metrics/plots/`: `master_bar_plots.png`, `master_radial_plot.png`,
  `master_bar_plots_1col.png` (paper), `master_bar_plots_ltm_cmab.png` (LTM vs
  CMAB), and `master_bar_plots_distributional[_no1q].png` (the 5-model
  DQN/RN-1q/RN distributional comparison).
- **`generate_legacy_summary.py`** — Run `legacy_simulation.py` over all 1000 UEs
  and write `results/final_metrics/legacy_baseline_summary.csv`.

## Summary / aggregation helpers
- **`aggregate.py`** — Post-training aggregator for parallel multi-seed benchmarks.
- **`aggregate_overnight.py`** — Aggregate the overnight HP-search runs into a
  single comparison CSV.
- **`headline_to_summary.py`** — Collapse a multi-seed bmk dir's per-seed eval
  summaries into one `*_summary.csv`.
- **`summarize_bmk.py`** — Print a one-line summary of a bmk dir (overnight log).

## Single-trajectory / policy figures
- **`plot_policy_comparison.py`** — Head-to-head comparison of two handover
  policies on ONE UE trajectory (`--a`/`--b` from ltm, dqn, rn, softstep, ra1q,
  ra; `--single` for one policy alone; `--animate` for the MP4).
- **`gen_head_to_head_set.py`** — Generate the standard presentation set of
  head-to-head comparisons (4 pairs × 3 UEs, PNG + MP4).
- **`gen_single_policy_set.py`** — Generate the single-policy presentation set
  (one algorithm by itself).

## Distributional / aggregate figures
- **`plot_return_density.py`** — Return-distribution DENSITY (PDF) view of a
  QR-DQN agent's per-action values.
- **`plot_return_distributions.py`** — Does the distributional agent *see* the
  risk? (return spread vs. radio state).
- **`plot_risk_frontier.py`** — The risk frontier: reward vs. tail failures,
  three ways to dial risk aversion.
- **`plot_per_ue_tails.py`** — Per-UE tail analysis: does risk aversion help the
  *worst* users?
- **`plot_finals_learning_overlay.py`** — Overlay the training-reward curves of
  the canonical no-gate final agents.
- **`plot_headline_curves.py`** — Training-reward curves comparing the key
  headline variants.
- **`plot_quantile_compare.py`** — Side-by-side learned-quantile comparison:
  midpoint baseline vs. cvar_trunc.

## Parameter / mode studies (plots)
- **`plot_quantile_mode_study.py`** — Aggregate and plot the LTM quantile-mode
  study (auto-discovers `bmk_*_qmode_*`).
- **`plot_atari_study.py`** — Aggregate and plot the Atari quantile-mode study.
- **`plot_cvar_alpha_sweep.py`** — CVaR α sweep (reward vs. HOF) for the paper.
- **`plot_kappa_sweep.py`** — Huber κ sweep (reward vs. HOF) for the paper.
- **`plot_n_sweep.py`** — N (num_quantiles) sweep: midpoint vs. cvar_truncated.

## Conceptual / talk figures
Slide-ready, transparent PNGs in the deck style; outputs go to `figures/`
(git-ignored) and are reproduced from these scripts.
- **`plot_qrdqn_explainer.py`** — Schematic network-head diagrams (DQN, QR-DQN,
  RN-1q, RA-1q, RA-full, RA-truncated) with `R^88` input and `a_1..a_N` labels.
- **`plot_quantile_representation.py`** — Representing a return distribution by N
  quantiles: PDF→quantiles, inverse-CDF staircase, CVaR tail, risk motivation.
- **`plot_pinball_loss.py`** — The pinball (quantile) loss and its Huber variant.

## Multi-config orchestrators (sweep configs under `configs/`)
- **`run_ablation.py`** — Sweep QR-DQN over `num_quantiles ∈ {10, 50, 100, 200}`.
- **`run_cvar_study.py`** — Sweep QR-DQN CVaR risk fractions on top of the
  quantile-ablation base.
- **`run_quantile_mode_study.py`** — Sweep QR-DQN quantile-positioning modes
  (midpoint / Gauss-Legendre / trapezoidal / CVaR-full / CVaR-truncated) on the
  LTM env.
- **`run_hp_refresh.py`** — Focused HP refresh (lr, τ, hard-update) after the
  target-net + QR-quadrature fixes.
- **`run_atari_study.py`** — The same quantile-mode sweep on Atari.

## Profiling
- **`benchmark_device.py`** — CPU vs. CUDA vs. XPU iteration-rate benchmark for
  the env-only and env+agent loops.
