# Overnight Massive Replications and Hyperparameter Search Sweep Report

| Configuration | Metrics Path | Reward/Decision | Capacity | Reliability (%) | HO Rate (/min) | HOF Rate (/min) | PP Rate (/min) | RLF Rate (/min) | Prep Rate | Res. Reservation (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Scalar DQN Baseline (`dqn.yaml`) | `bmk_2026-07-18_2_scalar-dqn-baseline` | 3.605±0.067 | 4.042 | 96.71% | 7.54±0.66 | 1.11±0.20 | 2.05±0.35 | 0.058 | 32.0 | 8.90% |
| Risk-Neutral QR-DQN N=1 (`rn_1q.yaml`) | `bmk_2026-07-18_3_risk-neutral-qr-dqn-n-1` | 3.593±0.034 | 4.041 | 96.55% | 9.32±1.07 | 1.30±0.21 | 3.22±0.75 | 0.062 | 33.9 | 8.72% |
| Risk-Neutral QR-DQN N=25 (`rn.yaml`) | `bmk_2026-07-18_4_risk-neutral-qr-dqn-n-25` | 3.602±0.080 | 4.049 | 96.72% | 7.67±0.76 | 1.06±0.19 | 2.33±0.32 | 0.064 | 32.8 | 8.97% |
| Soft-step (r=2.0) (`softcvar_r20.yaml`) | `bmk_2026-07-18_5_soft-step-r-2.0` | 3.595±0.047 | 4.045 | 96.71% | 7.26±0.65 | 0.96±0.09 | 2.10±0.49 | 0.060 | 31.8 | 9.03% |
| RA-1q (Dedicated Quantile) (`ra_1q.yaml`) | `bmk_2026-07-18_6_ra-1q-dedicated-quantile` | 3.577±0.130 | 4.032 | 96.64% | 6.86±0.71 | 0.81±0.10 | 2.05±0.41 | 0.071 | 30.4 | 8.97% |
| CVaR risk-dial a=0.5 (`cvarfull_a050.yaml`) | `bmk_2026-07-18_7_cvar-risk-dial-a-0.5` | 3.580±0.065 | 4.048 | 96.75% | 6.81±0.62 | 0.95±0.12 | 1.83±0.36 | 0.059 | 32.4 | 9.31% |
| CVaR risk-dial a=0.1 (`cvarfull_a010.yaml`) | `bmk_2026-07-18_8_cvar-risk-dial-a-0.1` | 3.542±0.038 | 4.028 | 96.76% | 5.75±0.66 | 0.77±0.14 | 1.37±0.33 | 0.069 | 32.2 | 9.77% |
| Quantile mode: Gauss-Legendre (`qmode_gauss_legendre.yaml`) | `bmk_2026-07-18_9_quantile-mode:-gauss-legendre` | 3.606±0.047 | 4.051 | 96.82% | 6.68±0.54 | 0.88±0.11 | 1.78±0.31 | 0.063 | 31.5 | 9.04% |
| Quantile mode: Trapezoidal (`qmode_trapezoidal.yaml`) | `bmk_2026-07-18_10_quantile-mode:-trapezoidal` | 3.595±0.049 | 4.048 | 96.78% | 6.55±0.43 | 0.86±0.11 | 1.65±0.27 | 0.058 | 31.2 | 9.05% |
| Quantile mode: Simpson (`qmode_simpson.yaml`) | `bmk_2026-07-18_11_quantile-mode:-simpson` | 3.600±0.063 | 4.045 | 96.72% | 7.21±0.61 | 1.00±0.14 | 1.96±0.34 | 0.069 | 31.5 | 8.89% |
| Quantile mode: Beta-equal (`qmode_beta_equal.yaml`) | `bmk_2026-07-18_12_quantile-mode:-beta-equal` | 3.593±0.029 | 4.049 | 96.65% | 8.26±0.45 | 1.13±0.08 | 2.66±0.31 | 0.065 | 33.3 | 8.94% |
| Quantile mode: Beta-weighted (`qmode_beta_weighted.yaml`) | `bmk_2026-07-18_13_quantile-mode:-beta-weighted` | 3.594±0.069 | 4.047 | 96.69% | 7.24±0.34 | 0.93±0.08 | 2.09±0.23 | 0.065 | 31.4 | 8.95% |
| Truncated CVaR a=0.5 (`trunc_a050.yaml`) | `bmk_2026-07-19_1_truncated-cvar-a-0.5` | 3.572±0.041 | 4.034 | 96.71% | 6.73±0.34 | 0.81±0.10 | 2.11±0.18 | 0.069 | 31.3 | 9.37% |
| Truncated CVaR a=0.25 (`trunc_a025.yaml`) | `bmk_2026-07-19_2_truncated-cvar-a-0.25` | 3.519±0.089 | 4.000 | 96.59% | 6.64±1.07 | 0.90±0.21 | 1.81±0.56 | 0.068 | 33.1 | 9.56% |
| Truncated CVaR a=0.1 (`trunc_a010.yaml`) | `bmk_2026-07-19_3_truncated-cvar-a-0.1` | 3.455±0.142 | 3.945 | 96.30% | 6.62±1.18 | 0.84±0.13 | 1.84±0.72 | 0.092 | 32.0 | 9.57% |
| Truncated CVaR a=0.75 (`trunc_a075.yaml`) | `bmk_2026-07-19_4_truncated-cvar-a-0.75` | 3.586±0.063 | 4.043 | 96.68% | 7.48±0.65 | 1.00±0.17 | 2.38±0.27 | 0.066 | 33.1 | 9.23% |
| HP Ablation: LR 2e-4 (`hp_lr2e4.yaml`) | `bmk_2026-07-19_5_hp-ablation:-lr-2e-4` | 3.599±0.076 | 4.043 | 96.69% | 7.22±0.39 | 1.00±0.17 | 1.95±0.04 | 0.063 | 32.0 | 8.96% |
| HP Ablation: LR 5e-5 (`hp_lr5e5.yaml`) | `bmk_2026-07-19_6_hp-ablation:-lr-5e-5` | 3.585±0.043 | 4.046 | 96.75% | 7.22±0.81 | 0.90±0.14 | 2.19±0.48 | 0.063 | 31.7 | 9.05% |
| HP Ablation: Gamma 0.95 (`hp_gamma095.yaml`) | `bmk_2026-07-19_7_hp-ablation:-gamma-0.95` | 3.568±0.039 | 4.025 | 96.44% | 10.31±0.69 | 1.35±0.14 | 3.95±0.53 | 0.066 | 36.0 | 8.73% |
| HP Ablation: Kappa 1.0 (`hp_kappa1.yaml`) | `bmk_2026-07-19_8_hp-ablation:-kappa-1.0` | 3.603±0.032 | 4.049 | 96.72% | 7.09±0.65 | 0.98±0.16 | 1.97±0.35 | 0.065 | 31.4 | 8.94% |
| DQN Width=128 (`dqn.yaml`) | `bmk_2026-07-19_9_dqn-width-128` | 3.600±0.081 | 4.047 | 96.68% | 8.07±1.08 | 1.05±0.18 | 2.56±0.77 | 0.063 | 33.0 | 8.98% |
| DQN Width=512 (`dqn.yaml`) | `bmk_2026-07-19_10_dqn-width-512` | 3.588±0.038 | 4.032 | 96.63% | 7.68±0.25 | 1.18±0.15 | 2.03±0.23 | 0.057 | 33.2 | 8.95% |
| DQN Width=1024 (`dqn.yaml`) | `bmk_2026-07-19_11_dqn-width-1024` | 3.549±0.102 | 4.017 | 96.53% | 7.10±0.97 | 1.13±0.28 | 1.67±0.38 | 0.065 | 34.7 | 9.29% |
| RN Width=128 (`rn.yaml`) | `bmk_2026-07-20_1_rn-width-128` | 3.592±0.033 | 4.047 | 96.76% | 7.26±0.83 | 0.92±0.24 | 2.13±0.35 | 0.062 | 32.0 | 9.03% |
| RN Width=512 (`rn.yaml`) | `bmk_2026-07-20_2_rn-width-512` | 3.595±0.034 | 4.042 | 96.73% | 6.57±0.20 | 0.90±0.05 | 1.65±0.12 | 0.067 | 30.8 | 8.96% |
| RN Width=1024 (`rn.yaml`) | `bmk_2026-07-20_3_rn-width-1024` | 3.582±0.055 | 4.040 | 96.75% | 6.15±0.43 | 0.87±0.10 | 1.33±0.15 | 0.069 | 32.5 | 9.30% |
| RA-1q Width=128 (`ra_1q.yaml`) | `bmk_2026-07-20_4_ra-1q-width-128` | 3.590±0.062 | 4.037 | 96.65% | 7.06±0.78 | 0.87±0.25 | 2.09±0.35 | 0.068 | 29.7 | 8.81% |
| Risk-Neutral QR-DQN N=25 (Width=512) (`rn.yaml`) | `bmk_2026-07-21_2_risk-neutral-qr-dqn-n-25-width-512` | 3.584±0.029 | 4.044 | 96.79% | 5.95±0.68 | 0.77±0.15 | 1.37±0.44 | 0.077 | 33.4 | 9.49% |
| Soft-step (r=2.0, Width=512) (`softcvar_r20.yaml`) | `bmk_2026-07-21_3_soft-step-r-2.0,-width-512` | 3.564±0.056 | 4.041 | 96.81% | 5.82±0.50 | 0.74±0.07 | 1.38±0.29 | 0.084 | 35.7 | 9.84% |
| RA-1q (Dedicated Quantile, Width=512) (`ra_1q.yaml`) | `bmk_2026-07-22_1_ra-1q-dedicated-quantile,-width-512` | 3.526±0.059 | 4.015 | 96.66% | 6.74±0.77 | 0.95±0.16 | 1.83±0.50 | 0.103 | 38.1 | 10.06% |
| CVaR risk-dial a=0.25 (Width=512) (`cvarfull_a025.yaml`) | `bmk_2026-07-22_2_cvar-risk-dial-a-0.25-width-512` | 3.518±0.040 | 4.008 | 96.68% | 5.73±0.78 | 0.79±0.10 | 1.36±0.53 | 0.102 | 36.9 | 10.22% |
| CVaR risk-dial a=0.1 (Width=512) (`cvarfull_a010.yaml`) | `bmk_2026-07-22_3_cvar-risk-dial-a-0.1-width-512` | 3.477±0.072 | 3.984 | 96.64% | 5.86±0.67 | 0.83±0.18 | 1.42±0.31 | 0.091 | 39.2 | 10.65% |
| DQN Hybrid Size-LR Limit (12k) (Inf-1) (`dqn.yaml`) | `bmk_2026-07-23_1_dqn-hybrid-size-lr-limit-12k-inf-1` | 3.501±0.095 | 3.992 | 96.39% | 8.44±1.38 | 1.42±0.27 | 2.37±0.81 | 0.081 | 41.5 | 9.80% |
| RN Hybrid Size-LR Limit (12k) (Inf-1) (`rn.yaml`) | `bmk_2026-07-23_2_rn-hybrid-size-lr-limit-12k-inf-1` | 3.564±0.036 | 4.030 | 96.66% | 6.71±0.67 | 0.87±0.05 | 1.88±0.60 | 0.080 | 34.8 | 9.43% |
| RA-1q Hybrid Size-LR Limit (12k) (Inf-1) (`ra_1q.yaml`) | `bmk_2026-07-24_1_ra-1q-hybrid-size-lr-limit-12k-inf-1` | 3.507±0.035 | 4.006 | 96.61% | 6.45±0.95 | 0.91±0.19 | 1.81±0.58 | 0.113 | 38.8 | 10.27% |
| DQN Hybrid Size-Batch Limit (12k) (Inf-1) (`dqn.yaml`) | `bmk_2026-07-24_2_dqn-hybrid-size-batch-limit-12k-inf-1` | 3.528±0.104 | 4.009 | 96.53% | 7.22±0.65 | 1.17±0.17 | 1.74±0.29 | 0.080 | 38.0 | 9.72% |
| RN Hybrid Size-Batch Limit (12k) (Inf-1) (`rn.yaml`) | `bmk_2026-07-24_3_rn-hybrid-size-batch-limit-12k-inf-1` | 3.580±0.040 | 4.038 | 96.75% | 6.23±0.71 | 0.84±0.17 | 1.60±0.41 | 0.082 | 34.5 | 9.49% |
| DQN Hybrid Size-LR Limit (12k) (Inf-2) (`dqn.yaml`) | `bmk_2026-07-25_1_dqn-hybrid-size-lr-limit-12k-inf-2` | 3.501±0.095 | 3.992 | 96.39% | 8.44±1.38 | 1.42±0.27 | 2.37±0.81 | 0.081 | 41.5 | 9.80% |
| RN Hybrid Size-LR Limit (12k) (Inf-2) (`rn.yaml`) | `bmk_2026-07-25_2_rn-hybrid-size-lr-limit-12k-inf-2` | 3.564±0.036 | 4.030 | 96.66% | 6.71±0.67 | 0.87±0.05 | 1.88±0.60 | 0.080 | 34.8 | 9.43% |
| RA-1q Hybrid Size-LR Limit (12k) (Inf-2) (`ra_1q.yaml`) | `bmk_2026-07-25_3_ra-1q-hybrid-size-lr-limit-12k-inf-2` | 3.507±0.035 | 4.006 | 96.61% | 6.45±0.95 | 0.91±0.19 | 1.81±0.58 | 0.113 | 38.8 | 10.27% |
| DQN Hybrid Size-Batch Limit (12k) (Inf-2) (`dqn.yaml`) | `bmk_2026-07-26_1_dqn-hybrid-size-batch-limit-12k-inf-2` | 3.528±0.104 | 4.009 | 96.53% | 7.22±0.65 | 1.17±0.17 | 1.74±0.29 | 0.080 | 38.0 | 9.72% |
