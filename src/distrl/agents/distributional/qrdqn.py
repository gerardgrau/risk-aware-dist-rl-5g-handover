"""Quantile Regression DQN agent.

The default configuration (quantile_mode='midpoint', risk_type='mean') is
exactly vanilla QR-DQN. The agent additionally supports non-uniform quantile
positioning (Gauss-Legendre, trapezoidal-with-fixed-endpoints) and a
"truncated" CVaR variant that only learns the bottom-k quantiles. See
`quantile_modes.py` for details.
"""

import os
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from src.distrl.agents.base import BaseAgent
from src.distrl.agents.distributional.quantile_modes import (
    QuantileScheme,
    build_scheme,
)
from src.distrl.agents.networks import (
    QuantileHead,
    UnifiedQNet,
    build_trunk,
)


class QRDQNAgent(BaseAgent):
    """Quantile Regression DQN with pluggable quantile-positioning schemes."""

    def __init__(
        self,
        config: dict[str, Any],
        observation_space: Any,
        action_space: Any,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, observation_space, action_space, device)

        action_dim = action_space.n

        self.num_quantiles = int(config.get("num_quantiles", 25))
        self.kappa = float(config.get("kappa", 1.0))
        self.risk_type = config.get("risk_type", "mean")
        self.risk_fraction = float(config.get("risk_fraction", 0.1))
        self.quantile_mode = config.get("quantile_mode", "midpoint")
        self.truncate_upper = bool(
            config.get("truncate_upper_quantiles", False)
        )
        self.dense_truncate = bool(config.get("dense_truncate", False))
        self.cvar_soft_ratio = float(config.get("cvar_soft_ratio", 0.0))
        self.single_quantile_action = bool(
            config.get("single_quantile_action", False)
        )
        self.density_ratio = float(config.get("density_ratio", 2.0))
        self.q_min = float(config.get("q_min", 0.0))
        self.q_max = float(config.get("q_max", 50.0))
        self.beta_alpha = float(config.get("beta_alpha", 2.0))
        self.beta_beta = float(config.get("beta_beta", 2.0))

        self.scheme: QuantileScheme = build_scheme(
            mode=self.quantile_mode,
            num_quantiles=self.num_quantiles,
            device=self.device,
            q_min=self.q_min,
            q_max=self.q_max,
            risk_type=self.risk_type,
            risk_fraction=self.risk_fraction,
            truncate_upper=self.truncate_upper,
            dense_truncate=self.dense_truncate,
            cvar_soft_ratio=self.cvar_soft_ratio,
            single_quantile_action=self.single_quantile_action,
            density_ratio=self.density_ratio,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
        )

        def make_qnet() -> UnifiedQNet:
            trunk = build_trunk(config, observation_space)
            head = QuantileHead(
                trunk.output_dim, action_dim, self.scheme.num_predicted,
            )
            return UnifiedQNet(trunk, head).to(self.device)

        # Independent module instances so the soft-update copy is real
        # (not a shared-reference no-op).
        self.q_net = make_qnet()
        self.target_net = make_qnet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.adam_eps = float(config.get("adam_eps", 1e-8))
        gc = config.get("grad_clip", None)
        self.grad_clip = float(gc) if gc is not None else None

        # Fused Adam fuses the param update into a single kernel — big
        # win on accelerators, CPU's foreach path is already competitive.
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=float(config.get("lr", 1e-4)),
            eps=self.adam_eps,
            fused=self.device.type != "cpu",
        )

    def _action_values(self, predicted: torch.Tensor) -> torch.Tensor:
        """Aggregate predicted quantiles into per-action Q-values.

        Args:
            predicted: shape [..., A, num_predicted].
        Returns:
            Shape [..., A].
        """
        if self.risk_type == "cvar":
            return self.scheme.cvar(predicted)
        return self.scheme.expectation(predicted)

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        valid_mask: np.ndarray | None = None,
    ) -> int:
        if np.random.rand() < epsilon:
            if valid_mask is not None:
                return int(np.random.choice(np.flatnonzero(valid_mask)))
            return int(self.action_space.sample())

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            predicted = self.q_net(state_t)  # [1, A, num_predicted]
            q_values = self._action_values(predicted).squeeze(0)  # [A]
            if valid_mask is not None:
                m = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~m, float("-inf"))
            return int(q_values.argmax().item())

    def train_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        states, actions, rewards, next_states, dones, next_valid_mask = batch
        num_predicted = self.scheme.num_predicted

        with torch.no_grad():
            next_predicted = self.target_net(next_states)  # [B, A, P]
            next_values = self._action_values(next_predicted)  # [B, A]
            # Restrict the greedy next action to valid (prepared) targets;
            # all-True mask (Atari) leaves this unchanged.
            next_values = next_values.masked_fill(~next_valid_mask, float("-inf"))
            best_next_actions = next_values.argmax(dim=1)
            best_next_predicted = next_predicted.gather(
                1,
                best_next_actions.view(-1, 1, 1).expand(-1, 1, num_predicted),
            ).squeeze(1)  # [B, P]
            # assemble_full prepends q_min / appends q_max for trapezoidal;
            # no-op for other schemes.
            best_next_assembled = self.scheme.assemble_full(
                best_next_predicted
            )  # [B, T]
            target_quantiles = (
                rewards + (1 - dones) * self.gamma_n * best_next_assembled
            )  # [B, T]

        current_quantiles = self.q_net(states).gather(
            1, actions.unsqueeze(-1).expand(-1, -1, num_predicted),
        ).squeeze(1)  # [B, P]

        # Pinball-Huber matrix over (predictor i, target j).
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        abs_diff = diff.abs()
        huber_loss = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )
        tau = self.scheme.tau.view(1, -1, 1)
        asymmetry = torch.abs(tau - (diff.detach() < 0).float())
        pinball_loss = asymmetry * huber_loss  # [B, P, T]

        # Quadrature-weighted reduction. For midpoint both weight vectors
        # are uniform 1/N and this reduces to a plain .mean(); for non-
        # uniform schemes (Gauss-Legendre, trapezoidal) the weights ARE
        # the integration rule — using .mean() would silently discard it.
        target_w = self.scheme.mean_weights.view(1, 1, -1)
        per_predictor_loss = (pinball_loss * target_w).sum(dim=2)  # [B, P]
        pred_w = self.scheme.predictor_weights.view(1, -1)
        loss = (per_predictor_loss * pred_w).sum(dim=1).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip,
            )
        self.optimizer.step()

        self.update_counter += 1
        self._update_target(self.q_net, self.target_net)

        # Detached 0-d tensor — caller materialises once per logging
        # interval to avoid a D2H sync per train step.
        return {"loss": loss.detach()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # Stash the scheme settings so checkpoints are self-describing.
                "scheme": {
                    "mode": self.scheme.mode,
                    "num_predicted": self.scheme.num_predicted,
                    "quantile_mode": self.quantile_mode,
                    "num_quantiles": self.num_quantiles,
                    "risk_type": self.risk_type,
                    "risk_fraction": self.risk_fraction,
                    "truncate_upper": self.truncate_upper,
                    "cvar_soft_ratio": self.cvar_soft_ratio,
                    "single_quantile_action": self.single_quantile_action,
                    "density_ratio": self.density_ratio,
                    "q_min": self.q_min,
                    "q_max": self.q_max,
                    "beta_alpha": self.beta_alpha,
                    "beta_beta": self.beta_beta,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
