import glob
import os
import re

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.distrl.envs.physics import (
    HO,
    NBS,
    ReceiverSensitivity,
    System,
    Time,
    Time_ContextRelease_11,
    Time_HOdecision_8,
    Time_LLHOCommand_9,
    Time_MeasReportL1_67,
    Time_MeasReportL3_1,
    Time_PingPong,
    Time_RA_10,
    Time_RRCConf3,
    Time_RRCReconf4_5,
    Time_RRCTransfer2,
    CheckHO_Failure,
    MCSEvaluation,
    physics_hash,
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


# Tick counts for each HO sub-phase. Precomputed so the simulation
# generator avoids re-running int(np.ceil(...)) on every iteration.
_TICK_MEAS_L3 = int(np.ceil(Time_MeasReportL3_1 / Time["TimeStep"]))
_TICK_RRC_PREP = int(np.ceil(
    (Time_RRCTransfer2 + Time_RRCConf3) / Time["TimeStep"]
))
_TICK_RRC_RECONF = int(np.ceil(Time_RRCReconf4_5 / Time["TimeStep"]))
_TICK_MEAS_L1_HODEC = int(np.ceil(
    (Time_MeasReportL1_67 + Time_HOdecision_8) / Time["TimeStep"]
))
_TICK_LL_HOCMD = int(np.ceil(Time_LLHOCommand_9 / Time["TimeStep"]))
_TICK_RA = int(np.ceil(Time_RA_10 / Time["TimeStep"]))
_TICK_CTX_RELEASE = int(np.ceil(Time_ContextRelease_11 / Time["TimeStep"]))
_TICK_PINGPONG = int(np.floor(Time_PingPong / Time["TimeStep"]))

# Reference inter-decision span (ticks) for the optional time-weighted reward.
# Each decision's reward is scaled by span/_SPAN_REF so the episode return is
# the cadence-invariant time-integral of penalized throughput. Set to the
# empirical mean inter-decision span (~no-gate event-driven) so the average
# reward magnitude matches the legacy per-decision mean-MCS reward — i.e. the
# A/B isolates the time-weighting effect, not a reward-scale change. Measured:
# mean inter-decision span ≈ 44 ticks (no-gate event-driven, random-valid policy).
_SPAN_REF = 44

_STATE_MAP = {'NORMAL_STEP': 0, 'FIND_CELL': 1, 'HO_DECISION': 2}


class LTMEnv(gym.Env):
    def __init__(self, config=None):
        super(LTMEnv, self).__init__()
        self.config = config or {}

        self.output_base = self.config.get('paths', {}).get('precomputed_data_directory', "data/Precomputed_v2")
        self.subdir = self.config.get('paths', {}).get('dataset_subdir', "no_ris")
        self.data_dir = os.path.join(self.output_base, self.subdir)
        if not os.path.exists(self.data_dir):
            if os.path.exists(os.path.join("data/Precomputed", self.subdir)):
                self.output_base = "data/Precomputed"
                self.data_dir = os.path.join(self.output_base, self.subdir)
            elif os.path.exists(os.path.join("data/sample", self.subdir)):
                self.output_base = "data/sample"
                self.data_dir = os.path.join(self.output_base, self.subdir)

        all_files = sorted(glob.glob(os.path.join(self.data_dir, "User*_precomputed.npz")), key=natural_sort_key)
        ue_count = self.config.get('simulation', {}).get('ue_number', len(all_files))
        if ue_count == 0 or ue_count > len(all_files):
            ue_count = len(all_files)
        self.files = all_files[:ue_count]
        self.current_ue_idx = 0

        self.NBS = NBS
        self.Max_iter = int(Time["TotalSimTime"] / Time["TimeStep"])

        # Action space: Target BS to HO to (or -1 for stay/none)
        self.action_space = spaces.Discrete(self.NBS)

        # Observation Space: 88-dim Markovian vector (Paper Aligned)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(88,), dtype=np.float32)

        # Cache per-step config lookups.
        self._p_window = int(self.config.get('ho_state', {}).get(
            'moving_average_window', 50,
        ))
        ho_reward_cfg = self.config.get('ho_reward', {})
        self._alpha_ho = float(ho_reward_cfg.get('alpha_ho', 0.8))
        self._alpha_pp = float(ho_reward_cfg.get('alpha_pp', 0.9))
        self._alpha_hof = float(ho_reward_cfg.get('alpha_hof', 0.1))

        # Event-driven RL cadence. When true, env.step advances to the NEXT
        # HO_DECISION so the agent acts on every handover decision (per-decision
        # semi-MDP), instead of a fixed 10-cycle (~1 s) bundle. Off by default →
        # legacy bundled cadence and baseline parity are untouched. Temporary
        # A/B toggle; hardcode if adopted.
        self._event_driven = bool(
            self.config.get('simulation', {}).get('event_driven', False)
        )

        # Learned-trigger ("no-gate") mode. When true, the agent is offered a
        # decision at every cycle where >=1 PREPARED cell other than serving
        # exists (not gated by the +3 dB exec rule), and may execute a handover
        # to ANY prepared cell — i.e. the RL replaces the +3 dB execution trigger
        # entirely with its learned value (preparation stays the LTM heuristic;
        # handovers still restricted to prepared cells). Off by default → the
        # standard LTM +3 dB trigger gates decisions. Implies event-driven cadence.
        self._learned_trigger = bool(
            self.config.get('simulation', {}).get('learned_trigger', False)
        )

        # Time-weighted reward (A/B). When true, each decision's reward is
        # scaled by its inter-decision span (in _SPAN_REF units) so the episode
        # return becomes the cadence-invariant time-integral of penalized
        # throughput — removing the per-decision-count bias that lets short
        # (stay) cycles over-accumulate. Off by default → legacy per-decision
        # mean-MCS reward. The composite_reward METRIC is reported regardless.
        self._time_weighted_reward = bool(
            self.config.get('simulation', {}).get('time_weighted_reward', False)
        )

        # Reusable buffer for _get_rl_obs (returned as .copy() each call).
        self._obs_buf = np.empty(88, dtype=np.float32)

        # O(1) running sums for the per-sector MA used in the observation.
        self._ma_last_t: int | None = None
        self._mcs_sum = np.zeros(self.NBS)
        self._snir_sum = np.zeros(self.NBS)

        # Per-episode bookkeeping arrays — allocated once, zeroed in-place
        # by reset(). The reset path reallocates iff a loaded .npz has a
        # different Max_iter (not expected for the canonical dataset).
        self._allocate_episode_buffers(self.Max_iter)

    def _allocate_episode_buffers(self, max_iter: int) -> None:
        self.ServingBSSector = np.full(max_iter, -1, dtype=int)
        self.MCS = np.zeros(max_iter)
        self.HOF = np.zeros(max_iter)
        self.RLF = np.zeros(max_iter)
        self.HO_event = np.zeros(max_iter)
        self.ping_pong = np.zeros(max_iter)
        self.ReservedBSSectors = np.zeros((self.NBS, max_iter))
        self.ListBSPrepared = np.zeros(self.NBS, dtype=int)
        self.TimerEntering = np.zeros(self.NBS)
        self.TimerLeaving = np.zeros(self.NBS)

    def _reset_episode_buffers(self) -> None:
        self.ServingBSSector.fill(-1)
        self.MCS.fill(0)
        self.HOF.fill(0)
        self.RLF.fill(0)
        self.HO_event.fill(0)
        self.ping_pong.fill(0)
        self.ReservedBSSectors.fill(0)
        self.ListBSPrepared.fill(0)
        self.TimerEntering.fill(0)
        self.TimerLeaving.fill(0)

    def _get_obs_dict(self, t, state_type='NORMAL_STEP', HO_condition=None, PL1_report=None):
        """Per-yield observation dict consumed by env.step.

        Only the fields actually used for `state_type` are populated:
        NORMAL_STEP reads only `state_type` + `t`; FIND_CELL adds the
        channel snapshot; HO_DECISION adds the L1 report + HO mask.
        """
        obs = {
            "state_type": _STATE_MAP[state_type],
            "state_str": state_type,
            "t": t,
        }
        if state_type == 'FIND_CELL':
            obs["ChBS2UE_t"] = self.ChBS2UE[:, t]
        elif state_type == 'HO_DECISION':
            obs["PL1_report"] = PL1_report
            obs["HO_condition"] = HO_condition
        return obs

    def _advance_ma(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-sector window averages of MCS / SNIR up to time `t`.

        Maintains running sums over the trailing `_p_window` ticks
        (slid forward in-place between consecutive calls) so the cost
        per call is O(advance) rather than O(window).
        """
        p_window = self._p_window
        new_start = max(0, t - p_window + 1)
        n = (t + 1) - new_start

        if self._ma_last_t is None or t < self._ma_last_t:
            # First call after reset (or t walked back) — rebuild from scratch.
            self._mcs_sum = self.all_mcs_episode[:, new_start:t + 1].sum(axis=1)
            self._snir_sum = self.all_snir_episode[:, new_start:t + 1].sum(axis=1)
            self._ma_last_t = t
            return self._mcs_sum / n, self._snir_sum / n

        if t == self._ma_last_t:
            return self._mcs_sum / n, self._snir_sum / n

        add_lo = self._ma_last_t + 1
        self._mcs_sum += self.all_mcs_episode[:, add_lo:t + 1].sum(axis=1)
        self._snir_sum += self.all_snir_episode[:, add_lo:t + 1].sum(axis=1)
        old_start = max(0, self._ma_last_t - p_window + 1)
        if new_start > old_start:
            self._mcs_sum -= self.all_mcs_episode[:, old_start:new_start].sum(axis=1)
            self._snir_sum -= self.all_snir_episode[:, old_start:new_start].sum(axis=1)
        self._ma_last_t = t
        return self._mcs_sum / n, self._snir_sum / n

    def _get_rl_obs(self) -> np.ndarray:
        """88-dim Markovian observation: speed, tenure, one-hot serving,
        RSRP[NBS], MA(MCS)[NBS], MA(SNIR)[NBS], (x, y)."""
        t = min(self.t, self.Max_iter - 1)
        avg_mcs, avg_snir = self._advance_ma(t)

        nbs = self.NBS
        buf = self._obs_buf

        buf[0] = self.ue_speeds[t] / 30.0
        buf[1] = float(self.serving_tenure) / 1000.0

        buf[2:2 + nbs] = 0.0
        s = self.ServingBSSector[t]
        if s != -1:
            buf[2 + s] = 1.0

        buf[2 + nbs:2 + 2 * nbs] = (self.PL3[:, t] + 75) / 45
        buf[2 + 2 * nbs:2 + 3 * nbs] = avg_mcs / 9.3
        buf[2 + 3 * nbs:2 + 4 * nbs] = (avg_snir - 15) / 25.0

        buf[2 + 4 * nbs] = self.ue_positions[t, 0] / 500.0
        buf[2 + 4 * nbs + 1] = self.ue_positions[t, 1] / 500.0

        # Caller holds `state` and `next_state` simultaneously — return a
        # copy so consecutive env.step returns don't alias the buffer.
        return buf.copy()

    def valid_action_mask(self) -> np.ndarray:
        """Boolean [NBS] selection mask: the prepared cells plus the serving
        cell (stay). The agent restricts its action selection to this set, so
        it can only target a prepared candidate. The prepared set is used here
        (rather than the instantaneous +3 dB HO_condition) because it is stable
        across the 100 ms RL step, whereas HO_condition is re-evaluated every
        10 ms tick; the +3 dB execution condition is enforced at the true
        decision tick by the generator's HO gate, so a selected cell that does
        not meet +3 dB there simply becomes a stay. Always has >= 1 True
        entry."""
        mask = self.ListBSPrepared.astype(bool).copy()
        s = self.ServingBSSector[min(self.t, self.Max_iter - 1)]
        if s != -1:
            mask[s] = True
        if not mask.any():
            mask[:] = True  # no serving + nothing prepared: action is ignored
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        filename = self.files[self.current_ue_idx % len(self.files)]
        self.current_ue_idx += 1

        with np.load(filename) as data:
            cached_hash = str(data['physics_hash'].item()) if 'physics_hash' in data.files else None
            current_hash = physics_hash()
            if cached_hash != current_hash:
                raise RuntimeError(
                    f"Precomputed cache {filename} has physics_hash="
                    f"{cached_hash!r} but current physics is {current_hash!r}. "
                    "Rebuild with: "
                    "`./venv-RL/bin/python3 src/tools/preprocess_dataset.py`"
                )
            self.ChBS2UE = data['ch_bs2ue']
            self.all_mcs_episode = data['all_mcs_episode']
            self.all_snir_episode = data['all_snir_episode']
            self.all_pe_episode = data['all_pe_episode']
            self.ue_positions = data['ue_positions']
            self.PL1 = data['pl1']
            self.PL3 = data['pl3']

        loaded_max = self.ChBS2UE.shape[1]
        if loaded_max != self.Max_iter:
            self.Max_iter = loaded_max
            self._allocate_episode_buffers(loaded_max)
        else:
            self._reset_episode_buffers()

        self.ue_speeds = np.full(self.Max_iter, 10.0)
        self.Sync = {
            "N310": 4, "N311": 2, "T310": 50,
            "out_sync_count": 0, "in_sync_count": 0,
            "t310_running": False, "t310_counter": np.inf
        }
        self.t = 0
        self.serving_tenure = 0
        self._ep_score_sum = 0.0   # Σ span·(mean-MCS reward) → composite_reward.
        self._ep_util_sum = 0.0    # Σ span·(MCS·reliability) → util_throughput.
        self._ep_span_ticks = 0    # Σ span lengths covered by decisions.
        self._ep_n_decisions = 0   # # decisions (env.step) → per-action mean reward.
        self._ma_last_t = None  # Invalidate the running-MA cache.

        self._sim_gen = self._simulation_generator()
        self._last_obs_dict = next(self._sim_gen)
        if self._event_driven:
            # Hand the agent its first HO_DECISION, not the initial FIND_CELL.
            self._skip_to_decision()
        return self._get_rl_obs(), {"metrics": self._get_obs_dict(0)}

    def _greedy_recovery_action(self, obs_dict: dict) -> int:
        """Env-side greedy cell (re)acquisition used at FIND_CELL yields."""
        ch = obs_dict['ChBS2UE_t']
        if float(np.max(ch)) + System["TxPower"] > ReceiverSensitivity:
            return int(np.argmax(ch))
        return -1

    def _skip_to_decision(self) -> bool:
        """Auto-advance the generator through FIND_CELL/NORMAL_STEP yields until
        it pauses at an HO_DECISION. Returns True if the episode terminated
        first. Event-driven cadence: the agent only ever sees HO_DECISION states.
        """
        while self._last_obs_dict['state_type'] != _STATE_MAP['HO_DECISION']:
            if self._last_obs_dict['state_type'] == _STATE_MAP['FIND_CELL']:
                act = self._greedy_recovery_action(self._last_obs_dict)
            else:  # NORMAL_STEP
                act = -1
            try:
                self._last_obs_dict = self._sim_gen.send(act)
            except StopIteration:
                return True
        return False

    def step(self, action, high_res_callback=None):
        if high_res_callback is not None:
            return self._step_high_res(high_res_callback)
        if self._event_driven:
            return self._step_event_driven(action)
        return self._step_bundled(action)

    def _finish_step(self, t_start: int, terminated: bool):
        """Shared tail: reward over [t_start, t_end), obs, info."""
        reward = 0.0
        if not terminated:
            # Reward aggregates over [t_start, t_end) rather than the latest
            # yielded tick — MCS/HO/PP/HOF for tick t are written at the TOP of
            # the NEXT outer iter, so sampling the just-yielded t reads a zero.
            t_end = min(self._last_obs_dict['t'], self.Max_iter)
            canonical, throughput = self._compute_step_reward(t_start, t_end)
            span = max(t_end - t_start, 1)
            # Duration-weighted accumulation for the cadence-invariant
            # composite_reward metric (tracked regardless of the training form).
            self._ep_score_sum += span * canonical
            # Throughput-only accumulation (MCS x reliability, no event
            # multipliers) for the ranking metric U; events are applied by COUNT
            # (per-min rates) post-hoc, avoiding the duration-weighting quirk.
            self._ep_util_sum += span * throughput
            self._ep_span_ticks += span
            self._ep_n_decisions += 1
            # Training reward: legacy per-decision mean-MCS, or (flag) the
            # span-weighted time-integral in _SPAN_REF units.
            reward = (canonical * span / _SPAN_REF
                      if self._time_weighted_reward else canonical)
        info = {}
        if terminated:
            info["metrics"] = self._build_metrics_dict()
        return self._get_rl_obs(), reward, terminated, False, info

    def _step_high_res(self, high_res_callback):
        """Baseline-parity cadence: callback fires on every tick, 10-tick bundle."""
        terminated = False
        t_start = self._last_obs_dict['t']
        for _ in range(10):
            try:
                act = high_res_callback(self._last_obs_dict)
                self._last_obs_dict = self._sim_gen.send(act)
            except StopIteration:
                terminated = True
                break
        return self._finish_step(t_start, terminated)

    def _step_bundled(self, action):
        """Legacy RL cadence: one action consumed at the first HO_DECISION in a
        10-cycle bundle; env auto-handles FIND_CELL recovery; NORMAL_STEP ignores
        action. (Preserved for backward compatibility; off the event-driven path.)
        """
        terminated = False
        t_start = self._last_obs_dict['t']
        for _ in range(10):
            try:
                current_state = self._last_obs_dict['state_type']
                if current_state == _STATE_MAP['FIND_CELL']:
                    self._last_obs_dict = self._sim_gen.send(
                        self._greedy_recovery_action(self._last_obs_dict))
                elif current_state == _STATE_MAP['HO_DECISION']:
                    self._last_obs_dict = self._sim_gen.send(action)
                    action = -1
                else:  # NORMAL_STEP
                    self._last_obs_dict = self._sim_gen.send(-1)
            except StopIteration:
                terminated = True
                break
        return self._finish_step(t_start, terminated)

    def _step_event_driven(self, action):
        """Per-decision cadence: apply the agent's action to the current
        HO_DECISION, then advance to the next one (auto-handling recovery /
        non-decision cycles). Reward accumulates over the inter-decision span."""
        t_start = self._last_obs_dict['t']
        terminated = False
        try:
            self._last_obs_dict = self._sim_gen.send(action)
        except StopIteration:
            terminated = True
        if not terminated:
            terminated = self._skip_to_decision()
        return self._finish_step(t_start, terminated)

    def _compute_step_reward(self, t_start: int, t_end: int) -> tuple[float, float]:
        """Multiplicative reward over the simulator tick range
        [t_start, t_end).

        Returns (reward, throughput) where `throughput` = avg_MCS x reliability
        (the continuous half, WITHOUT the discrete HO/PP/HOF event multipliers).
        `throughput` feeds the cadence-clean ranking metric U (events applied by
        count, not duration); `reward` is the full per-decision training reward.
        """
        if t_end <= t_start:
            return 0.0, 0.0
        avg_mcs = float(self.MCS[t_start:t_end].mean())
        multiplier = 1.0
        if np.any(self.HO_event[t_start:t_end] > 0):
            multiplier *= self._alpha_ho
        if np.any(self.ping_pong[t_start:t_end] > 0):
            multiplier *= self._alpha_pp
        if np.any(self.HOF[t_start:t_end] > 0):
            multiplier *= self._alpha_hof
        n_oos = self.Sync["out_sync_count"]
        reliability_factor = 1.0 / (1.0 + np.exp(2 * (n_oos - 2)))
        throughput = avg_mcs * reliability_factor
        return throughput * multiplier, throughput

    def _simulation_generator(self):
        t = 0
        NextBSSector = -1
        former_BS = -1
        former_HO_time = -np.inf

        while t < (self.Max_iter - 10):
            self.t = t
            self.ReservedBSSectors[:, t] = self.ListBSPrepared

            # Mirror legacy line 321: `if ServingBSSector[t] >= 0 and not RLF[t]`.
            # Using `NextBSSector` here would evaluate one extra MCS at the
            # post-HO RACH tick (where ServingBSSector is still -1 from init),
            # consuming an extra RNG call and desyncing all subsequent
            # stochastic interference draws.
            if self.ServingBSSector[t] >= 0 and not self.RLF[t]:
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    NextBSSector = -1

            t += 1
            self.t = t

            # Legacy Line 317: while NextBSSector == -1:
            while NextBSSector == -1 and t < self.Max_iter - 1:
                action = yield self._get_obs_dict(t, 'FIND_CELL')
                if action != -1:
                    NextBSSector = action
                    self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(NextBSSector, self.ChBS2UE[:, t], System, self.Sync)
                    if self.MCS[t] == 0:
                        NextBSSector = -1
                    else:
                        # Reset
                        self.Sync.update({"out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf})
                        self.serving_tenure = 0
                t += 1
                self.t = t

            # 5G HO logic
            self.ServingBSSector[t] = NextBSSector
            self.serving_tenure += 1
            self.ReservedBSSectors[:, t] = self.ListBSPrepared
            self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
            
            if self.RLF[t]:
                NextBSSector = -1
                continue

            # 1. L3 Measurement report
            PL3_report = self.PL3[:, t]
            Tf = min(t + _TICK_MEAS_L3, self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    break
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # 2 & 3. Preparation (RRC transfer + config)
            Tf = min(t + _TICK_RRC_PREP, self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    break
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # Conditions
            In_condition = PL3_report > (PL3_report[self.ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
            self.TimerEntering = (self.TimerEntering + Time["TimeStep"]) * In_condition
            self.ListBSPrepared = np.logical_or(self.ListBSPrepared, (self.TimerEntering > HO["Prep"]["PreparationTime"]))

            Out_condition = PL3_report < (PL3_report[self.ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
            self.TimerLeaving = (self.TimerLeaving + Time["TimeStep"]) * Out_condition
            self.ListBSPrepared = np.logical_and(self.ListBSPrepared, ~(self.TimerLeaving > HO["Prep"]["PreparationTime"]))

            if np.sum(self.ListBSPrepared) > HO["Prep"]["MaxNumberPreparedBS"]:
                metric = (10 ** (PL3_report / 10)) * self.ListBSPrepared
                I_sorted = np.argsort(metric)[::-1]
                self.ListBSPrepared[I_sorted[HO["Prep"]["MaxNumberPreparedBS"]:]] = 0

            # 4 & 5. RRC reconfiguration
            Tf = min(t + _TICK_RRC_RECONF, self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    break
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # EXECUTION
            PL1_report = self.PL1[:, t]
            HO_condition = np.logical_and(self.ListBSPrepared, PL1_report > (PL1_report[self.ServingBSSector[t]] + HO["Prep"]["ExecPowerOffset"]))

            Tf = min(t + _TICK_MEAS_L1_HODEC, self.Max_iter-1)
            while t < Tf:
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                if self.RLF[t]:
                    break
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # Decision gate + execution-eligibility depend on the trigger mode:
            #  - default (LTM trigger): offer a decision only when a prepared cell
            #    beats serving by +3 dB (HO_condition); execute only to such cells.
            #  - learned_trigger (no-gate): offer a decision whenever >=1 prepared
            #    cell other than serving exists; the agent may execute to ANY
            #    prepared cell, replacing the +3 dB rule with its learned value.
            _s = self.ServingBSSector[t]
            if self._learned_trigger:
                _cand = self.ListBSPrepared.copy()
                if _s != -1:
                    _cand[_s] = 0
                _gate = bool(_cand.any())
                _exec_ok = self.ListBSPrepared
            else:
                _gate = np.sum(HO_condition) > 0
                _exec_ok = HO_condition

            if _gate:
                action = yield self._get_obs_dict(t, 'HO_DECISION', HO_condition, PL1_report)
                # A handover fires only if the agent picks a non-serving cell that
                # is execution-eligible: HO_condition (+3 dB) in LTM mode, or any
                # prepared cell in learned_trigger mode. Any other pick (serving,
                # or a non-eligible cell) is a stay — no HO event, no alpha_HO.
                if (action != -1
                        and action != _s
                        and _exec_ok[action]):
                    t0 = t
                    self.HO_event[t0] = 1
                    I = action
                    
                    Tf = min(t + _TICK_LL_HOCMD, self.Max_iter-1)
                    while t < Tf:
                        self.ReservedBSSectors[:, t] = self.ListBSPrepared
                        self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                        if self.RLF[t]:
                            break
                        t += 1
                        self.t = t
                        self.serving_tenure += 1
                        self.ServingBSSector[t] = self.ServingBSSector[t - 1]
                    if self.RLF[t]: NextBSSector = -1; continue

                    t = min(t + _TICK_RA, self.Max_iter-1)
                    self.t = t
                    self.serving_tenure += 1
                    self.HOF[t] = CheckHO_Failure(I, self.ChBS2UE[:, t], System)

                    if self.HOF[t] == 0:
                        NextBSSector = I
                        t = min(t + _TICK_CTX_RELEASE, self.Max_iter-1)
                        self.t = t
                        self.serving_tenure += 1

                        self.ReservedBSSectors[:, t0:t + 1] = np.tile(self.ListBSPrepared.reshape(-1, 1), (1, t - t0 + 1))
                        self.ListBSPrepared[NextBSSector] = 0
                        self.serving_tenure = 0
                    else:
                        NextBSSector = -1

                    self.ping_pong[t0] = (NextBSSector == former_BS) and ((t0 - former_HO_time) < _TICK_PINGPONG)
                    former_BS = self.ServingBSSector[t0]
                    former_HO_time = t0
            else:
                # NORMAL STEP — mirror legacy fall-through.
                # Legacy doesn't yield here; it just lets the outer while loop
                # re-enter at the same t, which writes ReservedBSSectors[:, t]
                # and runs MCSEvaluation again as the new iter's first eval.
                # We yield for parity with the RL action protocol (the agent
                # receives an observation each cycle), but we do NOT advance t
                # nor pre-compute MCS — those happen at the top of the next
                # outer iter, exactly like legacy.
                action = yield self._get_obs_dict(t, 'NORMAL_STEP')

    def _build_metrics_dict(self):
        return {
            "mcs": self.MCS,
            "rlf": self.RLF,
            "ho": self.HO_event,
            "hof": self.HOF,
            "pp": self.ping_pong,
            "serving": self.ServingBSSector,
            "pl3": self.PL3,
            "reserved": self.ReservedBSSectors,
            # Cadence-invariant time-average of penalized throughput over the
            # episode's decision spans (same ponderation as the reward). A
            # comparable scalar across agents/baseline (≠ episode reward sum).
            "composite_reward": self._ep_score_sum / max(self._ep_span_ticks, 1),
            # Throughput half of the ranking metric U: time-averaged
            # MCS×reliability (events excluded). U = util_throughput ×
            # α_HO^(HO/min)·α_PP^(PP/min)·α_HOF^(HOF/min), computed post-hoc.
            "util_throughput": self._ep_util_sum / max(self._ep_span_ticks, 1),
            # Decision count → "reward per action" = reward / n_decisions.
            "n_decisions": self._ep_n_decisions,
        }
