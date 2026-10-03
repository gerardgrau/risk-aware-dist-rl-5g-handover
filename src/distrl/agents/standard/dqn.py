import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.distrl.agents.base import BaseAgent
from src.distrl.agents.networks import QHead, UnifiedQNet, build_trunk


class DQNAgent(BaseAgent):
    """Standard Deep Q-Network Agent."""

    def __init__(
        self,
        config: dict[str, Any],
        observation_space: Any,
        action_space: Any,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, observation_space, action_space, device)

        action_dim = action_space.n

        def make_qnet() -> UnifiedQNet:
            trunk = build_trunk(config, observation_space)
            head = QHead(trunk.output_dim, action_dim)
            return UnifiedQNet(trunk, head).to(self.device)

        # Independent module instances so the soft-update copy is real
        # (not a shared-reference no-op).
        self.q_net = make_qnet()
        self.target_net = make_qnet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Fused Adam fuses the param update into a single kernel — big
        # win on accelerators, CPU's foreach path is already competitive.
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=float(config.get("lr", 1e-4)),
            fused=self.device.type != "cpu",
        )

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

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0)
            if valid_mask is not None:
                m = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~m, float("-inf"))
            return int(q_values.argmax().item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        states, actions, rewards, next_states, dones, next_valid_mask = batch
        rewards = rewards.squeeze(1)
        dones = dones.squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            # Restrict the bootstrap max to valid (prepared) next actions;
            # all-True mask (Atari) leaves this unchanged.
            next_q = next_q.masked_fill(~next_valid_mask, float("-inf"))
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma_n * max_next_q

        current_q = self.q_net(states).gather(1, actions).squeeze(1)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        self._update_target(self.q_net, self.target_net)

        # Detached 0-d tensor — caller materialises once per logging
        # interval to avoid a D2H sync per train step.
        return {"loss": loss.detach()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
