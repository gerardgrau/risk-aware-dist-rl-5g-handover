"""Quantile positioning schemes for QR-DQN.

QR-DQN parametrizes a return distribution by predicting its inverse CDF at a
set of quantile fractions tau_i in [0, 1]. The expected value is

    E[Z] = integral_0^1 F^{-1}(tau) d tau

which we approximate with a quadrature rule defined by (tau_i, w_i). The
default vanilla QR-DQN uses the midpoint rule (uniform tau, uniform w).
This module also supports:

  - midpoint       : tau_i = (i + 0.5) / N, w_i = 1/N  (vanilla QR-DQN)
  - gauss_legendre : Gauss-Legendre nodes on [0,1], non-uniform tau and w
  - trapezoidal    : uniform tau including endpoints; fixed q_min/q_max at
                     tau=0/1; network only predicts the N-2 interior points
  - simpson        : composite Simpson 1/3 rule on a uniform grid; fixed
                     q_min/q_max at tau=0/1; network only predicts the N-2
                     interior points; weights follow the 1:4:2:4:...:4:1
                     pattern. Requires N odd (so N-1 is even and the rule
                     composes over pairs of intervals). Order-4 convergence
                     vs midpoint/trapezoidal's order-2, but less aggressive
                     than Gauss-Legendre, which may help when the integrand
                     (a ReLU-network quantile function) is piecewise-linear
                     rather than smooth.
  - beta_equal     : "Risk-sensitive / outlier-blind". tau_i = Beta_icdf(u_i),
                     u_i uniform midpoints (so the tau_i are quantiles of
                     Beta(alpha, beta) -- center-clustered for alpha=beta=2).
                     Mean_weights and predictor_weights stay uniform 1/N on
                     purpose: the expectation is a plain mean over the
                     densely-central quantiles, so tail outcomes get the
                     same voting power as the many central ones -- breaking
                     the true expected value. The distorted tau still
                     enters the QR Huber asymmetry, so the loss "knows"
                     about the new placement.
  - beta_weighted  : "High-resolution". Same Beta-distorted tau, but
                     mean_weights = predictor_weights = F^{-1}(u_{i+1}) -
                     F^{-1}(u_i) (cell widths in tau-space, summing to 1).
                     This is the mathematically faithful midpoint quadrature
                     under the distorted grid: the network spends more
                     capacity at the center while action selection and loss
                     remain unbiased.

Risk-aware truncation (truncate_upper_quantiles): when used with the CVaR
risk policy, the network only predicts the bottom k = ceil(N * risk_fraction)
quantiles uniformly in [0, risk_fraction]. The upper part of the distribution
is never consulted at action time, so dropping it concentrates network
capacity on the part that actually matters for the risk policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import betaincinv


@dataclass
class QuantileScheme:
    """Concrete quantile placement and integration weights.

    Attributes:
        tau:                Quantile fractions used in the QR loss (only for
                            the network-predicted quantiles). Shape
                            [num_predicted].
        mean_weights:       Integration weights for the FULL target
                            distribution (predicted + fixed endpoints if
                            any). Sum to 1. Shape [num_total]. Used both
                            for E[Z] aggregation and as the target-axis
                            probability mass in the QR Huber loss.
        cvar_weights:       Integration weights for the bottom-risk_fraction
                            lobe (zero above the cutoff), renormalised so
                            they sum to 1 over the cutoff. Shape
                            [num_total]. None if no risk_fraction was
                            provided.
        predictor_weights:  Aggregation weights over the predictor-axis of
                            the QR loss (sums to 1). For midpoint /
                            gauss_legendre / truncate_upper this equals
                            mean_weights (predictor and target supports
                            coincide). For trapezoidal, predictor_weights
                            is uniform over the N-2 interior nodes (the
                            fixed endpoints appear in the target axis but
                            are not learned, so they get no predictor
                            weight). Shape [num_predicted].
        num_predicted:      How many quantile values the network's head
                            outputs.
        fixed_lo:           Value of F^{-1}(0). Only set for trapezoidal.
        fixed_hi:           Value of F^{-1}(1). Only set for trapezoidal.
        mode:               Human-readable name (for logging / checkpoints).
    """

    tau: torch.Tensor
    mean_weights: torch.Tensor
    cvar_weights: torch.Tensor | None
    predictor_weights: torch.Tensor
    num_predicted: int
    fixed_lo: float | None
    fixed_hi: float | None
    mode: str

    @property
    def num_total(self) -> int:
        return int(self.mean_weights.numel())

    @property
    def has_fixed_endpoints(self) -> bool:
        return self.fixed_lo is not None

    def assemble_full(self, predicted: torch.Tensor) -> torch.Tensor:
        """Prepend / append fixed endpoint values if this scheme uses them.

        Args:
            predicted: shape [..., num_predicted].

        Returns:
            Shape [..., num_total]. For schemes without fixed endpoints this
            is the input unchanged.
        """
        if not self.has_fixed_endpoints:
            return predicted
        lead = predicted.shape[:-1]
        lo = torch.full(
            (*lead, 1),
            float(self.fixed_lo),
            dtype=predicted.dtype,
            device=predicted.device,
        )
        hi = torch.full(
            (*lead, 1),
            float(self.fixed_hi),
            dtype=predicted.dtype,
            device=predicted.device,
        )
        return torch.cat([lo, predicted, hi], dim=-1)

    def expectation(self, predicted: torch.Tensor) -> torch.Tensor:
        """Quadrature estimate of E[Z] over the assembled distribution."""
        full = self.assemble_full(predicted)
        return (full * self.mean_weights).sum(dim=-1)

    def cvar(self, predicted: torch.Tensor) -> torch.Tensor:
        """Quadrature estimate of CVaR_{risk_fraction} over the assembled
        distribution. Falls back to expectation() if no cvar_weights were
        precomputed (this should not be called in that case)."""
        if self.cvar_weights is None:
            return self.expectation(predicted)
        full = self.assemble_full(predicted)
        return (full * self.cvar_weights).sum(dim=-1)


def _compute_cvar_weights(
    full_tau: torch.Tensor,
    full_weights: torch.Tensor,
    risk_fraction: float,
) -> torch.Tensor:
    """Build CVaR_{rf} weights from a full quadrature (tau, w).

    Mass is kept on the support points whose tau <= risk_fraction and
    renormalized to sum to 1.
    """
    mask = full_tau <= risk_fraction
    masked = torch.where(mask, full_weights, torch.zeros_like(full_weights))
    total = masked.sum()
    if total <= 0:
        # Degenerate (no support below risk_fraction): keep first node only.
        out = torch.zeros_like(full_weights)
        out[0] = 1.0
        return out
    return masked / total


def _compute_soft_step_cvar_weights(
    full_tau: torch.Tensor,
    full_weights: torch.Tensor,
    risk_fraction: float,
    ratio: float,
) -> torch.Tensor:
    """Soft two-step distortion weights for action aggregation.

    A finite relaxation of hard CVaR: quantiles at tau <= risk_fraction keep
    `ratio` times the base mass of those above the cutoff, then the result is
    renormalized to sum to 1. ratio -> inf recovers hard CVaR_{risk_fraction};
    ratio == 1 recovers the plain mean (RN). The positions stay uniform (all N
    quantiles remain active and faithfully trained) -- only the action-time
    aggregation is distorted, exactly as for the hard-CVaR weights.
    """
    below = full_tau <= risk_fraction
    scaled = torch.where(below, full_weights * ratio, full_weights)
    return scaled / scaled.sum()


def build_scheme(
    mode: str,
    num_quantiles: int,
    device: torch.device,
    *,
    q_min: float = 0.0,
    q_max: float = 50.0,
    risk_type: str = "mean",
    risk_fraction: float = 0.1,
    truncate_upper: bool = False,
    dense_truncate: bool = False,
    cvar_soft_ratio: float = 0.0,
    single_quantile_action: bool = False,
    density_ratio: float = 2.0,
    beta_alpha: float = 2.0,
    beta_beta: float = 2.0,
) -> QuantileScheme:
    """Construct a QuantileScheme from config knobs.

    Args:
        mode: 'midpoint' | 'gauss_legendre' | 'trapezoidal' | 'simpson' |
            'beta_equal' | 'beta_weighted'.
        num_quantiles: Total grid size N. For trapezoidal the network only
            predicts N-2 of these (the interior).
        device: Torch device for the returned tensors.
        q_min, q_max: Fixed endpoints used only in trapezoidal mode.
        risk_type: 'mean' or 'cvar'. Controls whether cvar_weights are built.
        risk_fraction: For risk_type='cvar', the bottom fraction of the
            distribution to integrate over.
        truncate_upper: If True (only valid with mode='midpoint' and
            risk_type='cvar'), drop the upper (1 - risk_fraction) of the
            quantile grid and place the remaining k = ceil(N * risk_fraction)
            quantiles uniformly in [0, risk_fraction].
        cvar_soft_ratio: If > 0 (only with mode='midpoint', risk_type='cvar'),
            replace the hard CVaR mask with a soft two-step distortion that
            gives quantiles below risk_fraction this multiple of the base mass
            of those above. 1.0 == plain mean, large == hard CVaR. 0.0 (the
            default) keeps the hard CVaR mask unchanged.
        single_quantile_action: If True (only with mode='midpoint',
            risk_type='cvar'), the FULL midpoint distribution is learned but
            action selection reads ONLY the single quantile nearest
            tau = risk_fraction/2 (a VaR-style point criterion). This decouples
            learning (full distribution -> representation) from acting (one
            low quantile -> sharp tail focus); choose num_quantiles so that the
            grid lands exactly on risk_fraction/2 (e.g. N=20 hits 0.125 for
            risk_fraction=0.25).
        density_ratio: For mode='step_dense', the ratio of lower-lobe to
            upper-lobe quantile *density* across the risk_fraction breakpoint.
        beta_alpha, beta_beta: Shape parameters of the Beta distribution used
            to distort tau in 'beta_equal' / 'beta_weighted'. alpha=beta=2.0
            gives a symmetric center-clustered grid (mode at tau=0.5).
    """
    if truncate_upper:
        if risk_type != "cvar":
            raise ValueError(
                "truncate_upper_quantiles=True only makes sense with "
                "risk_type='cvar'"
            )
        if mode != "midpoint":
            raise ValueError(
                "truncate_upper_quantiles=True is only supported with "
                f"quantile_mode='midpoint' (got {mode!r})"
            )
        # dense_truncate: spend ALL N quantiles inside [0, risk_fraction]
        # (k=N, high-resolution tail) instead of the usual k=ceil(N*rf).
        k = num_quantiles if dense_truncate else max(
            1, int(math.ceil(num_quantiles * risk_fraction)))
        tau_np = (np.arange(k) + 0.5) * risk_fraction / k
        w_np = np.full(k, 1.0 / k, dtype=np.float64)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        # CVaR over the full (truncated) support is the mean by construction,
        # so all three weight axes share the same tensor.
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=weights,
            predictor_weights=weights,
            num_predicted=k,
            fixed_lo=None,
            fixed_hi=None,
            mode="midpoint_truncated_dense" if dense_truncate else "midpoint_truncated",
        )

    if mode == "midpoint":
        n = num_quantiles
        tau_np = (np.arange(n) + 0.5) / n
        w_np = np.full(n, 1.0 / n, dtype=np.float64)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        if risk_type == "cvar":
            if single_quantile_action:
                # Learn the full distribution, but act on ONLY the single
                # quantile nearest tau = risk_fraction/2 (the RA-1q position).
                target = risk_fraction / 2.0
                idx = int(np.argmin(np.abs(tau_np - target)))
                oh = np.zeros(n, dtype=np.float64)
                oh[idx] = 1.0
                cvar_w = torch.tensor(oh, dtype=torch.float32, device=device)
                mode_name = "midpoint_var1q"
            elif cvar_soft_ratio > 0.0:
                cvar_w = _compute_soft_step_cvar_weights(
                    tau, weights, risk_fraction, cvar_soft_ratio)
                mode_name = "midpoint_softcvar"
            else:
                cvar_w = _compute_cvar_weights(tau, weights, risk_fraction)
                mode_name = "midpoint"
        else:
            cvar_w = None
            mode_name = "midpoint"
        # mean_weights and predictor_weights are uniform 1/n; share the
        # tensor since these are read-only quadrature constants.
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            predictor_weights=weights,
            num_predicted=n,
            fixed_lo=None,
            fixed_hi=None,
            mode=mode_name,
        )

    if mode == "gauss_legendre":
        if num_quantiles < 2:
            raise ValueError("gauss_legendre requires num_quantiles >= 2")
        nodes_np, w_np = np.polynomial.legendre.leggauss(num_quantiles)
        tau_np = (nodes_np + 1.0) / 2.0
        w_np = w_np / 2.0
        # Sort by tau (leggauss returns them ordered, but be explicit).
        order = np.argsort(tau_np)
        tau_np, w_np = tau_np[order], w_np[order]
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        cvar_w = (
            _compute_cvar_weights(tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        # GL has predictor_weights == mean_weights (same nodes/weights are
        # used for the outer integral over tau).
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            predictor_weights=weights,
            num_predicted=num_quantiles,
            fixed_lo=None,
            fixed_hi=None,
            mode="gauss_legendre",
        )

    if mode == "simpson":
        n = num_quantiles
        if n < 3:
            raise ValueError("simpson requires num_quantiles >= 3")
        if n % 2 == 0:
            raise ValueError(
                "simpson requires odd num_quantiles (so N-1 intervals is "
                "even and the composite 1/3 rule applies over pairs of "
                f"intervals). Got {n}; try {n + 1} or {n - 1}."
            )
        full_tau_np = np.linspace(0.0, 1.0, n, dtype=np.float64)
        # Composite Simpson 1/3 integer weights: 1, 4, 2, 4, 2, ..., 4, 1.
        # Integer sum = 3(n-1), so normalised mean_weights sum to 1.
        int_w = np.ones(n, dtype=np.float64)
        int_w[1:-1:2] = 4.0  # odd indices (1, 3, ..., n-2)
        int_w[2:-1:2] = 2.0  # even interior (2, 4, ..., n-3)
        full_w_np = int_w / (3.0 * (n - 1))
        interior_tau_np = full_tau_np[1:-1]
        # Predictor-axis weights: each interior node's Simpson integer weight
        # renormalised over the interior. Interior sum = 3n - 5 (drop the two
        # endpoint 1s from the total 3(n-1) = 3n - 3).
        interior_int_w = int_w[1:-1]
        predictor_w_np = interior_int_w / interior_int_w.sum()
        full_tau = torch.tensor(full_tau_np, dtype=torch.float32, device=device)
        interior_tau = torch.tensor(
            interior_tau_np, dtype=torch.float32, device=device,
        )
        weights = torch.tensor(full_w_np, dtype=torch.float32, device=device)
        predictor_weights = torch.tensor(
            predictor_w_np, dtype=torch.float32, device=device,
        )
        cvar_w = (
            _compute_cvar_weights(full_tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        return QuantileScheme(
            tau=interior_tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            predictor_weights=predictor_weights,
            num_predicted=n - 2,
            fixed_lo=float(q_min),
            fixed_hi=float(q_max),
            mode="simpson",
        )

    if mode == "trapezoidal":
        n = num_quantiles
        if n < 3:
            raise ValueError("trapezoidal requires num_quantiles >= 3")
        # Full grid: tau_i = i / (n - 1)
        full_tau_np = np.linspace(0.0, 1.0, n, dtype=np.float64)
        # Trapezoidal weights on a uniform grid of n points over [0,1]:
        # interior = 1/(n-1), endpoints = 1/(2(n-1)). Sum = 1.
        full_w_np = np.full(n, 1.0 / (n - 1), dtype=np.float64)
        full_w_np[0] = full_w_np[-1] = 0.5 / (n - 1)
        # The network only predicts interior quantiles (tau_1, ..., tau_{n-2}).
        interior_tau_np = full_tau_np[1:-1]
        full_tau = torch.tensor(full_tau_np, dtype=torch.float32, device=device)
        interior_tau = torch.tensor(
            interior_tau_np, dtype=torch.float32, device=device,
        )
        weights = torch.tensor(full_w_np, dtype=torch.float32, device=device)
        # Predictor axis: only the N-2 interior nodes are learned. Their
        # trapezoidal weights are all 1/(n-1), so renormalising to sum to 1
        # gives a uniform 1/(n-2). Stored explicitly for clarity.
        predictor_w_np = np.full(n - 2, 1.0 / (n - 2), dtype=np.float64)
        predictor_weights = torch.tensor(
            predictor_w_np, dtype=torch.float32, device=device,
        )
        cvar_w = (
            _compute_cvar_weights(full_tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        return QuantileScheme(
            tau=interior_tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            predictor_weights=predictor_weights,
            num_predicted=n - 2,
            fixed_lo=float(q_min),
            fixed_hi=float(q_max),
            mode="trapezoidal",
        )

    if mode in ("beta_equal", "beta_weighted"):
        n = num_quantiles
        if n < 2:
            raise ValueError(f"{mode} requires num_quantiles >= 2")
        if beta_alpha <= 0.0 or beta_beta <= 0.0:
            raise ValueError(
                "Beta shape parameters must be > 0 "
                f"(got alpha={beta_alpha}, beta={beta_beta})"
            )
        # tau_i = F^{-1}(u_i), the Beta(alpha, beta) quantile function at
        # uniform midpoints u_i = (i + 0.5)/n. The tau_i are themselves
        # samples from Beta(alpha, beta), so they cluster around the mode
        # (0.5 when alpha=beta=2). scipy.special.betaincinv is the inverse
        # regularized incomplete beta function.
        u_mid_np = (np.arange(n) + 0.5) / n
        tau_np = betaincinv(beta_alpha, beta_beta, u_mid_np)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        if mode == "beta_equal":
            # Variant 1: keep weights uniform on purpose.
            w_np = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            # Variant 2: cell widths in tau-space, F^{-1}(u_{i+1}) -
            # F^{-1}(u_i) with u_i = i/n. They sum to F^{-1}(1) - F^{-1}(0)
            # = 1 - 0 = 1 exactly, regardless of (alpha, beta).
            u_edge_np = np.arange(n + 1) / n
            tau_edges = betaincinv(beta_alpha, beta_beta, u_edge_np)
            w_np = np.diff(tau_edges)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        cvar_w = (
            _compute_cvar_weights(tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            predictor_weights=weights,
            num_predicted=n,
            fixed_lo=None,
            fixed_hi=None,
            mode=mode,
        )

    if mode == "step_dense":
        # Risk via *position*: a piecewise-uniform grid that packs the lower
        # lobe [0, risk_fraction] `density_ratio` times more densely than the
        # upper lobe, with uniform 1/N weights. Action selection is a plain
        # mean over this tail-dense grid (cvar_weights stay None -> the agent's
        # expectation() path is used), so the tail bias is carried entirely by
        # where the quantiles sit -- the positional dual of midpoint+softcvar.
        n = num_quantiles
        a = float(risk_fraction)
        if not 0.0 < a < 1.0:
            raise ValueError(
                f"step_dense requires 0 < risk_fraction < 1 (got {a})"
            )
        if density_ratio <= 0.0:
            raise ValueError(
                f"step_dense requires density_ratio > 0 (got {density_ratio})"
            )
        # n_lo / a = density_ratio * n_hi / (1 - a),  n_lo + n_hi = N.
        n_lo = int(round(n * density_ratio * a / (density_ratio * a + 1.0 - a)))
        n_lo = min(max(n_lo, 1), n - 1)
        n_hi = n - n_lo
        tau_lo = (np.arange(n_lo) + 0.5) * a / n_lo
        tau_hi = a + (np.arange(n_hi) + 0.5) * (1.0 - a) / n_hi
        tau_np = np.concatenate([tau_lo, tau_hi])
        w_np = np.full(n, 1.0 / n, dtype=np.float64)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=None,
            predictor_weights=weights,
            num_predicted=n,
            fixed_lo=None,
            fixed_hi=None,
            mode="step_dense",
        )

    raise ValueError(f"Unknown quantile_mode: {mode!r}")
