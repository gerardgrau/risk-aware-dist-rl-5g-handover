import os
from typing import Any
import numpy as np
import torch

from src.distrl.agents.base import BaseAgent
from src.distrl.envs.physics import System, calculate_snir_matrix

class CMABAgent(BaseAgent):
    """Contextual Multi-Armed Bandit Agent using Kernel UCB."""

    def __init__(
        self,
        config: dict[str, Any],
        observation_space: Any,
        action_space: Any,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, observation_space, action_space, device)
        
        self.action_dim = action_space.n
        
        # CMAB specific parameters
        self.alpha = float(config.get("alpha", 0.3))
        self.gamma_reg = float(config.get("gamma_reg", 1e-2))
        self.window_size = int(config.get("window_size", 1000))
        self.kernel_type = config.get("kernel_type", "ntk")
        
        # QoE reward parameters
        self.bw_mhz = float(config.get("bw_mhz", 50.0))
        self.handover_cost_frac = float(config.get("handover_cost_frac", 0.2))
        self.pingpong_cost_frac = float(config.get("pingpong_cost_frac", 0.1))
        self.mcs_weight = float(config.get("mcs_weight", 0.3))
        self.rsrp_weight = float(config.get("rsrp_weight", 0.0))
        self.sigma_thr_mhz = float(config.get("sigma_thr_mhz", 80.0))
        self.min_rsrp = float(config.get("min_rsrp", -120.0))
        self.max_rsrp = float(config.get("max_rsrp", -50.0))
        
        # Environment reference for online updates
        self.env = None
        
        # Internal states for online updates
        self.last_t = None
        self.last_arm = None
        self.last_context = None
        self.last_state_type = None
        
        self.reset()

    def set_env(self, env: Any) -> None:
        """Sets reference to the Gymnasium environment for history access."""
        self.env = env

    def reset(self) -> None:
        """Resets the bandit memory at the start of each episode."""
        self.memory_X = None  # torch.Tensor of shape [M, d]
        self.memory_y = None  # torch.Tensor of shape [M, 1]
        self.Kinv = None      # torch.Tensor of shape [M, M]
        
        self.last_t = None
        self.last_arm = None
        self.last_context = None
        self.last_state_type = None

    def _get_context(self, obs: np.ndarray) -> np.ndarray:
        """Reconstructs the 6D context per arm from the 88D state vector or env history."""
        if self.env is not None:
            t = self.env.t
            rsrp = self.env.PL3[:, t]
            serving_cell = self.env.ServingBSSector[t]
            speed = self.env.ue_speeds[t]
            t_lastHO = float(self.env.serving_tenure)
            
            X_t = np.zeros((self.action_dim, 6), dtype=np.float32)
            for a in range(self.action_dim):
                # Instantaneous MCS matching MATLAB computeMCS (from cached episode data)
                mcs_a = self.env.all_mcs_episode[a, t]
                
                # 100ms average SNIR matching MATLAB computeSNIR (from cached episode data)
                avg_snir_a = self.env.all_snir_episode[a, max(0, t - 9) : t + 1].mean()
                
                X_t[a, 0] = rsrp[a]
                X_t[a, 1] = mcs_a
                X_t[a, 2] = avg_snir_a
                X_t[a, 3] = float(serving_cell)
                X_t[a, 4] = speed
                X_t[a, 5] = t_lastHO
        else:
            # Fallback for compatibility (e.g. if env reference is not set)
            speed = obs[0] * 30.0
            t_lastHO = obs[1] * 1000.0
            serving_one_hot = obs[2:23]
            serving_cell = np.argmax(serving_one_hot) if np.any(serving_one_hot > 0) else -1
            rsrp = obs[23:44] * 45.0 - 75.0
            avg_mcs = obs[44:65] * 9.3
            avg_snir = obs[65:86] * 25.0 + 15.0
            
            X_t = np.zeros((self.action_dim, 6), dtype=np.float32)
            for a in range(self.action_dim):
                X_t[a, 0] = rsrp[a]
                X_t[a, 1] = avg_mcs[a]
                X_t[a, 2] = avg_snir[a]
                X_t[a, 3] = float(serving_cell)
                X_t[a, 4] = speed
                X_t[a, 5] = t_lastHO
            
        # Z-score normalization (MATLAB's adjust_dim keeps all d features;
        # the +1e-8 guards near-zero variance without dropping columns,
        # which would cause dimension mismatches in the kernel memory).
        mu = np.mean(X_t, axis=0)
        sigma = np.std(X_t, axis=0, ddof=1)
        sigma_norm = sigma + 1e-8
        X_norm = (X_t - mu) / sigma_norm
        return X_norm

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Computes the Arc-cosine (degree 1) / NTK kernel matrix."""
        # Row normalization
        eps = 1e-12
        norm1 = X1.norm(p=2, dim=-1, keepdim=True) + eps
        norm2 = X2.norm(p=2, dim=-1, keepdim=True) + eps
        X1n = X1 / norm1
        X2n = X2 / norm2
        
        # Cosine similarity
        S = torch.matmul(X1n, X2n.T)
        S = torch.clamp(S, -1.0, 1.0)
        
        # Arc-cosine / NTK kernel formula
        pi = np.pi
        acosS = torch.acos(S)
        K = (S * (pi - acosS) + torch.sqrt(torch.clamp(1.0 - S**2, min=0.0))) / pi
        return K

    def qoe_reward(
        self,
        rsrp: float,
        snir: float,
        mcs: float,
        ho_flag: bool,
        hof_flag: bool,
        rlf_flag: bool,
        ping_flag: bool,
    ) -> float:
        """Computes the QoE reward matching qoe_reward_vSigmoid.m."""
        # PHY Shannon throughput
        snr_lin = 10.0 ** (snir / 10.0)
        thr_shannon = self.bw_mhz * np.log2(1.0 + snr_lin)
        
        # MCS throughput proxy
        thr_mcs = mcs * self.bw_mhz
        
        # Combined throughput
        thr = (1.0 - self.mcs_weight) * thr_shannon + self.mcs_weight * thr_mcs
        
        # Optional RSRP bonus
        if self.rsrp_weight > 0:
            rsrp_norm = (rsrp - self.min_rsrp) / (self.max_rsrp - self.min_rsrp)
            thr += self.rsrp_weight * self.sigma_thr_mhz * max(0.0, min(1.0, rsrp_norm))
            
        # Penalties
        if ho_flag:
            thr *= (1.0 - self.handover_cost_frac)
        if ping_flag:
            thr *= (1.0 - self.pingpong_cost_frac)
        if hof_flag:
            thr *= 0.1
            
        # Sigmoid factor
        r = thr / (1.0 + np.exp(2.0 * (float(rlf_flag) - 2.0)))
        return float(r)

    def update_kernel(self, x_new: torch.Tensor, r_t: float) -> None:
        """Updates kernel memory and Kinv incrementally (block matrix inversion)."""
        gamma = self.gamma_reg
        W = self.window_size
        
        r_tensor = torch.tensor([[r_t]], dtype=torch.float32, device=self.device)
        
        if self.memory_X is None:
            self.memory_X = x_new.clone()
            self.memory_y = r_tensor
            k_xx = self._kernel(x_new, x_new)
            self.Kinv = torch.inverse(k_xx + gamma)
        else:
            k_new = self._kernel(self.memory_X, x_new)
            k_xx = self._kernel(x_new, x_new)
            
            Kt_inv = self.Kinv
            b = k_new
            s = k_xx + gamma - torch.matmul(b.T, torch.matmul(Kt_inv, b))
            if s <= 0:
                s = torch.tensor([[1e-6]], dtype=torch.float32, device=self.device)
                
            K22 = 1.0 / s
            K11 = Kt_inv + K22 * torch.matmul(torch.matmul(Kt_inv, b), torch.matmul(b.T, Kt_inv))
            
            # Construct new Kinv block
            col = -K22 * torch.matmul(Kt_inv, b)
            row = col.T
            Kinv_new = torch.cat([
                torch.cat([K11, col], dim=1),
                torch.cat([row, K22], dim=1)
            ], dim=0)
            
            self.Kinv = Kinv_new
            self.memory_X = torch.cat([self.memory_X, x_new], dim=0)
            self.memory_y = torch.cat([self.memory_y, r_tensor], dim=0)
            
        # Apply sliding window
        if self.memory_X.size(0) > W:
            self.memory_X = self.memory_X[-W:].clone()
            self.memory_y = self.memory_y[-W:].clone()
            K_full = self._kernel(self.memory_X, self.memory_X)
            I_W = torch.eye(W, dtype=torch.float32, device=self.device)
            self.Kinv = torch.inverse(K_full + gamma * I_W)

    def _perform_online_update(self) -> None:
        """Processes previous transition reward and updates kernel memory selectively."""
        if self.env is None or self.last_t is None:
            return
            
        t_start = self.last_t
        t_end = self.env.t
        
        is_rlf = bool(np.any(self.env.RLF[t_start:t_end] > 0))
        is_find_cell = (self.last_state_type == "FIND_CELL")
        
        # In MATLAB, the kernel is ONLY updated on initial connection (FIND_CELL) 
        # or when a Radio Link Failure (RLF) occurs. Successful handovers are NOT updated.
        if is_find_cell or is_rlf:
            # Gather events over the decision span
            is_ho = bool(np.any(self.env.HO_event[t_start:t_end] > 0))
            is_pingpong = bool(np.any(self.env.ping_pong[t_start:t_end] > 0))
            is_hof = bool(np.any(self.env.HOF[t_start:t_end] > 0))
            
            # Throughput features
            mcs_avg = float(self.env.MCS[max(0, t_end - 10):t_end].mean())
            
            # SNIR average
            snir_matrix = self.env.all_snir_episode[:, max(0, t_end - 10):t_end]
            snir_avg = float(snir_matrix[self.last_arm, :].mean())
            
            # MATLAB passes ChBS2UE(arm, t) to qoe_reward as the 'rsrp' argument
            raw_channel_gain = float(self.env.ChBS2UE[self.last_arm, t_end - 1])
            
            # Calculate QoE reward
            r_t = self.qoe_reward(
                rsrp=raw_channel_gain,
                snir=snir_avg,
                mcs=mcs_avg,
                ho_flag=is_ho,
                hof_flag=is_hof,
                rlf_flag=is_rlf,
                ping_flag=is_pingpong
            )
            
            # Update kernel
            self.update_kernel(self.last_context, r_t)
            
        # Reset decision tracker
        self.last_t = None
        self.last_arm = None
        self.last_context = None
        self.last_state_type = None

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        valid_mask: np.ndarray | None = None,
    ) -> int:
        # Check and process the previous decision step's update
        self._perform_online_update()
        
        # Normal high-res callback check
        if isinstance(state, dict):
            # We are in high-res callback mode
            state_type = state.get("state_str", "NORMAL_STEP")
            if state_type == "FIND_CELL":
                # In initial connection: CMAB evaluates all arms
                obs_vector = self.env._get_rl_obs()
                valid_mask = None
            elif state_type == "HO_DECISION":
                # In handover decision: CMAB evaluates prepared arms
                obs_vector = self.env._get_rl_obs()
                valid_mask = state["HO_condition"].copy()
                valid_mask[self.env.ServingBSSector[self.env.t]] = True
            else:
                return -1
        else:
            obs_vector = state
            state_type = "NORMAL_STEP"
            
        # Reconstruct context
        X_t_np = self._get_context(obs_vector)
        X_t = torch.tensor(X_t_np, dtype=torch.float32, device=self.device)
        
        # Predict UCB values
        mu_pred = torch.zeros(self.action_dim, 1, dtype=torch.float32, device=self.device)
        sigma_pred = torch.ones(self.action_dim, 1, dtype=torch.float32, device=self.device)
        
        if self.memory_X is not None:
            # Vectorized Kernel UCB prediction
            K_vec = self._kernel(self.memory_X, X_t)  # [M, 21]
            K_xx = torch.diag(self._kernel(X_t, X_t)).unsqueeze(1)  # [21, 1]
            
            v = torch.matmul(K_vec.T, self.Kinv)  # [21, M]
            mu_pred = torch.matmul(v, self.memory_y)  # [21, 1]
            
            quad_term = torch.diag(torch.matmul(v, K_vec)).unsqueeze(1)  # [21, 1]
            sigma_pred = torch.sqrt(torch.clamp(K_xx - quad_term, min=0.0))
            
        gamma_reg = self.gamma_reg
        ucb_values = mu_pred + (self.alpha / np.sqrt(gamma_reg)) * sigma_pred
        ucb_values = ucb_values.squeeze(1).cpu().numpy()
        
        # Apply valid mask
        if valid_mask is not None:
            # Set invalid actions to a very low value
            ucb_values = np.where(valid_mask, ucb_values, -1e9)
            
        action = int(np.argmax(ucb_values))
        
        # Save decision variables for next step's update
        if self.env is not None:
            self.last_t = self.env.t
            self.last_arm = action
            self.last_context = X_t[action:action+1].clone()
            self.last_state_type = state_type
            
        return action

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        # No-op for contextual bandits in standard offline RL loop
        return {"loss": torch.tensor(0.0, device=self.device)}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
