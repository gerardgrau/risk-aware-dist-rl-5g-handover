# AGENTS.md

This file provides guidance to autonomous AI coding agents (Antigravity, Claude Code, Cursor, GitHub Copilot, etc.) working with code in this repository.

## Project Overview

Research project applying **Distributional Reinforcement Learning (DQN vs QR-DQN)** to optimize **5G Lower Layer Triggered Mobility (LTM)** handover decisions under fast fading channels.

- **Deployment**: Multi-sector 5G deployment (7 BS × 3 sectors = 21 sectors, NBS=21).
- **Dataset & Generalization Split**:
  - **Training Split**: 2,000 independent trajectories (`data/ChannelGains_v2` / `data/Precomputed_v2`) over 6,000 episodes across 5 seeds.
  - **Out-of-Domain Test Split**: 1,000 completely unseen trajectories (`data/ChannelGains` / `data/Precomputed`) over 300,000 evaluation steps ($\epsilon = 0.0$).
  - **Sample Dataset**: Lightweight subset (`<1 MB`) in `data/sample/no_ris/` for immediate smoke tests and verification without large downloads.
- **Journal Publication**: IEEE Transactions on Machine Learning in Communications and Networking (TMLCN), 2026. Manuscript sources and compiled PDF live under `paper_tmlcn/`.

This file is the single source of truth for architecture, commands, parity decisions, and repository policies.

---

## Repository & Git Protocol

1. **Strict Single-Commit Policy**:
   - The `main` branch must always contain **exactly 1 commit** (`git rev-list --count HEAD == 1`).
   - Every change must be committed using `git commit --amend` and force-pushed to `origin/main` (`git push --force origin main`).
2. **Zero Pull Requests**:
   - Do NOT open Pull Requests on GitHub (`gh pr create` should never be run).
   - The repository must have 0 open or closed Pull Requests (`gh pr list --state all == 0`).
3. **File Deletion Rule**:
   - Always use the `trash` command instead of `rm` when deleting files.

---

## Environment & Common Commands

The project uses a Python 3.10+ virtualenv at `venv-RL/`; dependencies are listed in `requirements.txt`. Always export `PYTHONPATH` before invoking any script:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 <script>            # use the venv binary explicitly
```

### Training & Benchmarks
```bash
# Train proposed champion (Soft-step QR-DQN, alpha=0.25, r=2.0)
./venv-RL/bin/python3 src/main.py --config configs/models/softstep.yaml --device cpu

# Train Risk-Neutral QR-DQN baseline (N=25)
./venv-RL/bin/python3 src/main.py --config configs/models/rn.yaml --device cpu

# Train Scalar DQN baseline
./venv-RL/bin/python3 src/main.py --config configs/models/dqn.yaml --device cpu

# Train key truncated ablations
./venv-RL/bin/python3 src/main.py --config configs/models/trunc_k03_a010.yaml --device cpu   # Best failure-reduction truncated model
./venv-RL/bin/python3 src/main.py --config configs/models/trunc_k07_a025.yaml --device cpu   # Throughput-preserving truncated model
```

> **Device Note**: Always use `--device cpu`. Because `main.py` self-parallelizes seeds across CPU cores and the networks are compact MLPs, multi-core CPU is substantially faster in wall-clock than the Intel iGPU (`xpu`) or GPU overhead on this workload.

### Single-Agent Evaluation (Frozen Weights, $\epsilon = 0.0$, 1,000 Unseen Users)
```bash
./venv-RL/bin/python3 src/evaluate_model.py \
    --agent qrdqn --model results/benchmarks/<run>/models/qrdqn_best.pth \
    --config configs/models/softstep.yaml --output results/final_metrics/qrdqn_softstep
```

### Dataset Acquisition & Preprocessing
```bash
# Automated download of raw channel matrices from open repository
./scripts/download_dataset.sh

# Preprocess raw .mat files into fast compressed .npz archives (auto-detects both splits)
./venv-RL/bin/python3 src/tools/preprocess_dataset.py
```

### Verification & Smoke Tests (Runnable Scripts)
```bash
./venv-RL/bin/python3 src/scripts/test_dqn_ltm.py                   # DQN end-to-end smoke test
./venv-RL/bin/python3 src/scripts/test_qrdqn_ltm.py                 # QR-DQN end-to-end smoke test
./venv-RL/bin/python3 src/scripts/test_quantile_modes.py            # Quantile mathematics & schemes
./venv-RL/bin/python3 src/scripts/test_metrics_calc.py              # 9 standardized KPIs correctness
./venv-RL/bin/python3 src/scripts/verify_simulation_parity.py       # LTM baseline gym vs paper parity
./venv-RL/bin/python3 src/scripts/test_env.py                       # Atari smoke (Breakout via ale-py)
```

### Master Publication Visualizations
```bash
./venv-RL/bin/python3 src/tools/generate_final_plots.py         # Master bar/radial plots into results/final_metrics/plots/
./venv-RL/bin/python3 src/tools/plot_risk_frontier.py          # Empirical risk-return frontiers
./venv-RL/bin/python3 src/tools/plot_return_density.py         # Action-conditional return density distributions
./venv-RL/bin/python3 src/tools/plot_quantile_mode_study.py    # Quantile placement mode comparisons
```

---

## Architecture & Codebase Design

### Two Parallel Simulators (Strict Parity)
- `src/distrl/envs/ltm_gym.py` — Gymnasium `LTMEnv` used by RL agents. RL steps at 100 ms cadence; physical simulation advances at 10 ms.
- `src/distrl/envs/legacy_simulation.py` — Standalone reference simulator for baseline LTM hardcoded heuristic and paper parity.
- `src/distrl/envs/physics.py` — **Single source of truth** for all SINR/MCS/HOF math and constants (`TxPower=25 dBm`, `NoiseLevel=-101 dBm`, `ExecPowerOffset=3.0 dB`, `MaxNumberPreparedBS=5`). Do not duplicate physics code elsewhere.

### Agent Framework (`src/distrl/agents/`)
- `base.py` — `BaseAgent` ABC with soft/hard target network update mechanics.
- `networks.py` — `MLPTrunk` (88-dim vector input), `CNNTrunk` (Nature-DQN 84×84×4 conv stack), and interchangeable heads (`QHead`, `QuantileHead`) wired through `UnifiedQNet`.
- `standard/dqn.py` — Vanilla Deep Q-Network.
- `standard/cmab.py` — Published online Contextual Multi-Armed Bandit (LinUCB).
- `standard/ltm_baseline.py` — Hardcoded 3GPP LTM heuristic.
- `distributional/quantile_modes.py` — `QuantileScheme` dataclass & factory supporting midpoint, Gauss-Legendre, trapezoidal, and upper-truncated quantile schemes.
- `distributional/qrdqn.py` — QR-DQN supporting standard risk-neutral expectation, strict Hard CVaR, upper-tail truncation, and smooth **Soft-step** distortion ($\alpha=0.25, r=2.0$).

### Observation Space (88 Dimensions)
`[speed(1), tenure(1), serving_one_hot(21), RSRP(21), MA_MCS(21), MA_SNIR(21), x(1), y(1)]`.
Moving averages are computed in $O(1)$ running-sum time.

### Reward Function
Multiplicative reward:
$$R = R_{\text{thr}} \cdot \alpha_{\text{HO}}^{\text{ind}_{\text{HO}}} \cdot \alpha_{\text{PP}}^{\text{ind}_{\text{PP}}} \cdot \alpha_{\text{HOF}}^{\text{ind}_{\text{HOF}}} \cdot \text{reliability\_factor}$$
Paper parameters: $\alpha_{\text{HOF}}=0.1$, $\alpha_{\text{HO}}=0.8$, $\alpha_{\text{PP}}=0.9$.

---

## Directory Organization

```
risk-aware-dist-rl-5g-handover/
├── AGENTS.md           # This document (guidance for AI agents)
├── README.md           # Public user-facing documentation & executive benchmark
├── configs/
│   ├── models/         # 10 canonical configs matching paper Table II
│   ├── archive/        # Archived historical hyperparameter sweeps
│   ├── atari/          # Atari validation configs
│   ├── config.yaml     # Global default configuration
│   └── README.md       # Config documentation
├── data/
│   └── sample/         # Lightweight (<1 MB) sample dataset for quick testing
├── docs/
│   ├── assets/         # Publication figures and master plots
│   └── slides.pdf      # Presentation slide deck
├── figures/            # Curated figure gallery and explainer diagrams
├── paper_tmlcn/        # IEEE TMLCN LaTeX manuscript and compiled PDF
├── results/
│   └── final_metrics/  # Master cross-seed CSV summaries, sweeps/, and plots/
├── scripts/            # Dataset acquisition scripts
└── src/
    ├── distrl/         # Core framework (agents, envs, networks, physics, utils)
    ├── scripts/        # Verification and test scripts
    ├── tools/          # Plotting and preprocessing utilities
    ├── main.py         # Main training orchestrator
    └── evaluate_model.py # Frozen-model evaluation script
```

---

## Things to Double-Check Before Modifying Code

- Any changes to `physics.py` directly alter **paper parity**. Current values follow the published benchmark (`TxPower=25 dBm`, `NoiseLevel=-101 dBm` over 20 MHz, `MaxNumberPreparedBS=5`, 26-step SINR table). Always run `src/scripts/verify_simulation_parity.py` after editing physics.
- The 88-dimensional state vector is fixed; any dimensionality change invalidates saved `.pth` weights.
- Always maintain exactly 1 commit on `main` and 0 Pull Requests.
