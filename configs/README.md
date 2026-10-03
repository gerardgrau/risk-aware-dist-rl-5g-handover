# Experiment Configurations (`configs/`)

This directory contains the experiment configurations used across training, evaluation, and hyperparameter sweeps.

## Primary Paper Models (`configs/models/`)

These configurations correspond directly to the models reported in Table II of the IEEE TMLCN 2026 paper (*Comprehensive Out-of-Domain Generalization Benchmark*):

| Configuration | Model Name | Description | Quantiles ($k$) | Policy / Objective |
| :--- | :--- | :--- | :---: | :--- |
| **`configs/models/softstep.yaml`** | **Soft-step QR-DQN** | **Proposed Champion** | 25 | Soft spectral tail weighting ($\alpha=0.25, r=2.0$) |
| **`configs/models/rn.yaml`** | **RN QR-DQN** | Risk-neutral distributional baseline | 25 | Uniform quantile average $\frac{1}{N}\sum \theta_i$ |
| **`configs/models/dqn.yaml`** | **Scalar DQN** | Standard value-based baseline | 1 | Expected Q-value $\mathbb{E}[Q(s,a)]$ |
| **`configs/models/cvar_a025.yaml`** | **Hard CVaR** | Risk-averse lower-tail policy | 25 | Lower-tail CVaR at $\alpha=0.25$ |
| **`configs/models/cvar_a010.yaml`** | **Hard CVaR** | Conservative lower-tail policy | 25 | Lower-tail CVaR at $\alpha=0.10$ |
| **`configs/models/trunc_k7.yaml`** | **Truncated CVaR** | Tail-restricted representation | 7 | Lower $k=7$ quantiles ($\alpha=0.25$) |
| **`configs/models/trunc_k3.yaml`** | **Truncated CVaR** | Tail-restricted representation | 3 | Lower $k=3$ quantiles ($\alpha=0.10$) |
| **`configs/models/ra_1q.yaml`** | **RA-1q** | Single dedicated quantile | 1 | Direct prediction at $\tau=0.125$ |
| **`configs/models/cmab.yaml`** | **CMAB** | Published Contextual Bandit | — | Online LinUCB (Moreno-Locubiche et al. 2024) |
| **`configs/models/ltm_baseline.yaml`** | **LTM Baseline** | Standard 3GPP heuristic | — | Rule-based +3 dB L1 execution gate |

### Quickstart Execution

To run any model with the multi-seed benchmark workflow:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 src/main.py --config configs/models/softstep.yaml --device cpu
```

## Exploratory & Sweep Subdirectories

- **`masked/no_gate/`**: Contains the earlier sweep exploration templates (`finals_n25/` and `ablations_n25/`).
- **`atari/`**: Configuration templates for Atari benchmark validation (Breakout, Boxing, Pong).
- **`alpha_v3/`, `cvar_alpha/`, `qmode_v3/`, `hp_refresh/`**: Archived hyperparameter search configs referenced by sweep tools in `src/tools/`.
