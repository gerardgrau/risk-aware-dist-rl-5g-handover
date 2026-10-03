import numpy as np
from typing import Any


def calculate_8_metrics(
    mcs_history: np.ndarray,
    rlf_history: np.ndarray,
    ho_history: np.ndarray,
    hof_history: np.ndarray,
    pp_history: np.ndarray,
    reserved_history: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute the 8 LTM metrics from one episode's tick-level histories.

    `reserved_history` is the (NBS, total_steps) binary grid of prepared
    sectors the simulator tracked live (env `info["metrics"]["reserved"]`),
    matching the reference's `ReservedBSSectors`.
    """
    total_steps = len(mcs_history)
    time_step = config['simulation']['time_step']
    minutes = (total_steps * time_step) / 60.0

    # 1. Capacity (avg MCS)
    avg_capacity = np.mean(mcs_history)

    # 2. RLF Rate
    rlf_rate = np.sum(rlf_history) / minutes if minutes > 0 else 0.0

    # 3. HO Rate
    ho_rate = np.sum(ho_history) / minutes if minutes > 0 else 0.0

    # 4. PP Rate
    pp_rate = np.sum(pp_history) / minutes if minutes > 0 else 0.0

    # 5. Reliability (%)
    reliability = 100.0 - (np.mean(mcs_history == 0) * 100.0)

    # 6. Preparation EVENTS per minute: each 0->1 transition of the reservation
    # grid (a sector newly entering the prepared list). This matches the legacy
    # MATLAB simulator's `Number_cell_preparations` count, with its `>= 0`
    # threshold corrected to `> 0` (`>= 0` counts every non-decreasing tick).
    # It is NOT occupancy -- res_reservation_pct is occupancy.
    reserved_int = reserved_history.astype(np.int8)
    prep_events = int(np.sum(np.diff(reserved_int, axis=1) > 0)) if total_steps > 1 else 0
    prep_rate = prep_events / minutes if minutes > 0 else 0.0

    # 7. Resource reservation: % occupancy of the NBS x total_steps grid.
    resource_reservation = (np.sum(reserved_history) / (reserved_history.shape[0] * total_steps)) * 100.0

    # 8. HOF Rate
    hof_rate = np.sum(hof_history) / minutes if minutes > 0 else 0.0

    return {
        "ho_rate": float(ho_rate),
        "hof_rate": float(hof_rate),
        "pp_rate": float(pp_rate),
        "capacity_avg": float(avg_capacity),
        "rlf_rate": float(rlf_rate),
        "reliability_pct": float(reliability),
        "prep_rate": float(prep_rate),
        "res_reservation_pct": float(resource_reservation),
        "total_steps": int(total_steps),
        "total_minutes": float(minutes),
    }
