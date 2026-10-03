"""Hardcoded LTM baseline agent.

Stateless greedy policy that mirrors `legacy_simulation.py`'s in-loop
decisions. Designed for high-resolution callback mode:

    env.step(0, high_res_callback=agent.select_action)

The env yields one observation dict per simulation tick (`FIND_CELL`,
`HO_DECISION`, or `NORMAL_STEP`) and the agent picks the matching greedy
action. Constructor / save / load / train_step exist so the agent can be
selected via `main.py --agents ltm_baseline` alongside DQN/QR-DQN, but
they are no-ops — there are no learnable parameters.
"""

from typing import Any

import numpy as np

from src.distrl.envs.physics import ReceiverSensitivity, System


class LTMBaselineAgent:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        observation_space: Any = None,
        action_space: Any = None,
        device: str = "cpu",
    ) -> None:
        # Args mirror DQNAgent/QRDQNAgent for symmetric construction
        # in main.py; this agent has no learnable state.
        pass

    def reset(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def train_step(self, batch: tuple) -> dict[str, float]:
        return {"loss": 0.0}

    def select_action(
        self, obs: Any, epsilon: float = 0.0, valid_mask: Any = None,
    ) -> int:
        # In high-res callback mode `obs` is the per-tick dict produced
        # by `LTMEnv._get_obs_dict`. The 100ms RL-step path passes the
        # 88-dim observation vector instead, in which case there is no
        # decision context — return 0 so the env's internal greedy
        # fallback handles HO/FIND_CELL.
        if not isinstance(obs, dict):
            return 0

        state_type = obs.get("state_str", "NORMAL_STEP")

        if state_type == "FIND_CELL":
            ch = obs["ChBS2UE_t"]
            best = int(np.argmax(ch))
            if ch[best] + System["TxPower"] > ReceiverSensitivity:
                return best
            return -1

        if state_type == "HO_DECISION":
            pl1 = obs["PL1_report"]
            ho_condition = obs["HO_condition"]
            metric = (10 ** (pl1 / 10.0)) * ho_condition
            if np.sum(metric) > 0:
                return int(np.argmax(metric))
            return -1

        return -1
