import hashlib
import math

import numpy as np

# Bumped whenever the SNIR/MCS/HOF formulas change in a way that invalidates
# the on-disk precomputed cache (constants alone aren't enough — the cache
# bakes in formula outputs, so a formula change must reject old .npz files).
# v1: pre-2026-05-19. v2: M*Ps fix applied to MCSEvaluation, CheckHO_Failure,
# and calculate_snir_matrix.
FORMULA_VERSION = "v2_M_signal"

# ============================================================
# SYSTEM CONFIGURATIONS
# ============================================================
ReceiverSensitivity = -95

System = {
    "TxPower": 25,  # dBm
    "NoiseLevel": -174 + 10 * np.log10(20 * 1e6),  # dBm (20 MHz bandwidth)
    "SINRThreshold": np.array([
        -np.inf, -3, -2, 0, 2, 4, 6, 7, 10, 12, 14, 16, 20,
        22, 24, 26, 28, 30, 32, 35, 38, 40, 42, 44, 46, 48
    ]),
    "SpectralEff": np.array([
        0, 0.24, 0.38, 0.60, 0.88, 1.18, 1.46, 1.70, 1.92,
        2.40, 2.92, 3.40, 3.60, 4.14, 4.74, 5.28, 5.58, 5.7,
        5.85, 5.92, 6.64, 7.12, 7.44, 7.50, 8.30, 9.30
    ])
}

Time = {
    "TotalSimTime": 300,
    "TimeStep": 10e-3
}

BS = {"Number": 7}
NBS = BS["Number"] * 3

HO = {
    "Prep": {
        "PeriodicityRSRPMeasurement": 20e-3,
        "AverageRSRPMeasument_NL1": 10,
        "kL3": 8,
        "alphaIIRfilter": 2 ** (-8 / 4),
        "PreparationPowerOffset": -3,
        "PreparationTime": 40e-3,
        "ExecPowerOffset": 3.0,
        "MaxNumberPreparedBS": 5
    }
}

# Time variables
Time_PingPong = 1 
Time_InitialSync_0 = 0.0115
Time_MeasReportL3_1 = 0.01
Time_RRCTransfer2 = 0.01
Time_RRCConf3 = 0.022
Time_RRCReconf4_5 = 0.02
Time_MeasReportL1_67 = 0.01
Time_HOdecision_8 = 0.01
Time_LLHOCommand_9 = 0.02
Time_RA_10 = 10e-3
Time_ContextRelease_11 = 20e-3

# ============================================================
# PHYSICS HASH
# ============================================================
def physics_hash() -> str:
    """Short hash of the physics constants that drive the precomputed cache.

    Precomputed `.npz` files are stamped with this value; `LTMEnv.reset()`
    cross-checks it on load and refuses to run if it doesn't match — the
    cache must be regenerated whenever TxPower / NoiseLevel / SINR table
    changes, otherwise the agent's observations and the runtime physics
    diverge silently.
    """
    h = hashlib.sha256()
    h.update(FORMULA_VERSION.encode())
    h.update(str(System["TxPower"]).encode())
    h.update(str(System["NoiseLevel"]).encode())
    h.update(System["SINRThreshold"].tobytes())
    h.update(System["SpectralEff"].tobytes())
    return h.hexdigest()[:16]


# ============================================================
# PHYSICS FUNCTIONS
# ============================================================
# Hot-path constants reused by every MCSEvaluation / CheckHO_Failure
# call (~130k per episode). Derived from the module-level `System`
# dict and the fixed M-fold interference model.
_M_INTERFERENCE = 3
_INV_M = 1.0 / _M_INTERFERENCE
_LOW_INTER_FACTOR = 10 ** (-1.5)
_NOISE_LINEAR = 10 ** (System["NoiseLevel"] / 10.0)

_BLER = np.array([
    1, 0.2617, 0.2370, 0.2103, 0.1828, 0.1558, 0.1302, 0.1067, 0.0859,
    0.0678, 0.0526, 0.0401, 0.0301, 0.0221, 0.0160, 0.0114, 0.0080,
    0.0056, 0.0038, 0.0025, 0.0017, 0.0011, 0.0007, 0.0004, 0.0003,
    0.0002, 0.0001
])
_SNR_LEVEL = np.array([
    -np.inf, -1.7609, -1.6609, -1.5609, -1.4609, -1.3609, -1.2609,
    -1.1609, -1.0609, -0.9609, -0.8609, -0.7609, -0.6609, -0.5609,
    -0.4609, -0.3609, -0.2609, -0.1609, -0.0609, 0.0391, 0.1391,
    0.2391, 0.3391, 0.4391, 0.5391, 0.6391, 0.7391
])


def _aggregate_signal_and_interference(serving_sector, channels, tx_power):
    """Return (Ps, AllInter) in linear power for the reuse-group at this tick.

    The reuse-group is the 1:3 stride starting at `serving_sector % 3`
    (7 sectors for NBS=21). The Python loop beats a numpy reduce on
    such a short array (numpy dispatch overhead dominates) and matches
    its bit-for-bit summation order.
    """
    Ps = 10 ** ((channels[serving_sector] + tx_power) / 10)
    AllInter = 0.0
    for k in range(serving_sector % 3, channels.shape[0], 3):
        AllInter += 10 ** ((channels[k] + tx_power) / 10.0)
    AllInter -= Ps
    return Ps, AllInter


def MCSEvaluation(serving_sector, channels, System, Sync):
    Ps, AllInter = _aggregate_signal_and_interference(
        serving_sector, channels, System["TxPower"],
    )
    M = _M_INTERFERENCE
    M_AllInter = M * AllInter
    if np.random.rand() < _INV_M:
        Inter_Noise = M_AllInter + _NOISE_LINEAR
    else:
        Inter_Noise = M_AllInter * _LOW_INTER_FACTOR + _NOISE_LINEAR

    # M*Ps in the numerator mirrors the M-fold scaling already applied
    # to the interference term. Dropping it (a previous calibration
    # accident) made SNIR ~4.77 dB pessimistic in noise-limited regimes.
    SNIR = 10 * math.log10(M * Ps) - 10 * math.log10(Inter_Noise)
    i = int(np.searchsorted(System["SINRThreshold"], SNIR, side="right")) - 1
    MCS = System["SpectralEff"][i] if i >= 0 else 0

    RLF = 0
    if SNIR <= 0:
        Sync["out_sync_count"] += 1
        Sync["in_sync_count"] = 0
        if Sync["out_sync_count"] >= Sync["N310"] and not Sync["t310_running"]:
            Sync["t310_running"] = True
            Sync["t310_counter"] = Sync["T310"]
    if SNIR > 2:
        Sync["in_sync_count"] += 1
        Sync["out_sync_count"] = 0
        if Sync["in_sync_count"] >= Sync["N311"]:
            Sync["t310_running"] = False
    if Sync["t310_running"]:
        Sync["t310_counter"] -= 1
        RLF = (Sync["t310_counter"] == 0)
    return MCS, RLF, Sync


def CheckHO_Failure(serving_sector, channels, System):
    Ps, AllInter = _aggregate_signal_and_interference(
        serving_sector, channels, System["TxPower"],
    )
    M = _M_INTERFERENCE
    M_AllInter = M * AllInter
    if np.random.rand() < _INV_M:
        Inter_Noise = M_AllInter + _NOISE_LINEAR
    else:
        Inter_Noise = M_AllInter * _LOW_INTER_FACTOR + _NOISE_LINEAR

    SNIR = 10 * math.log10(M * Ps) - 10 * math.log10(Inter_Noise)
    i = int(np.searchsorted(_SNR_LEVEL, SNIR, side="right")) - 1
    Pe = _BLER[i] if i >= 0 else _BLER[0]
    return np.random.rand() < Pe

def PerformanceEvaluation(MCS, ServingCell_channel, ping_pong, ReservedBSSectors, HOF, HO_event, RLF, Time):
    Minutes = MCS.shape[0] * Time["TimeStep"] / 60
    Performance = {
        "Capacity": MCS,
        "RL_problems": np.sum(RLF) / Minutes,
        "Number_HO": np.sum(HO_event) / Minutes,
        "Number_ping_pongs": np.sum(ping_pong) / Minutes,
        "Reliability": 100 - np.mean(MCS == 0) * 100,
        "Number_cell_preparations": np.sum((ReservedBSSectors[:, 1:] - ReservedBSSectors[:, :-1]) >= 0) / Minutes,
        "Resource_reservation": np.sum(ReservedBSSectors) / ReservedBSSectors.size * 100,
        "HOF": np.sum(HOF) / Minutes
    }
    return Performance

def calculate_snir_matrix(channels_2d: np.ndarray, system_params: dict) -> np.ndarray:
    nbs, total_t = channels_2d.shape
    tx_power = system_params.get("TxPower", 25)
    noise_level = system_params.get("NoiseLevel", -174)
    
    ps_linear = 10 ** ((channels_2d + tx_power) / 10.0)
    group_sums = np.zeros((3, total_t))
    for g in range(3):
        group_sums[g, :] = np.sum(ps_linear[g::3, :], axis=0)
        
    all_sectors = np.arange(nbs)
    all_inter_linear = group_sums[all_sectors % 3, :] - ps_linear

    M = 3
    noise_floor = 10**(noise_level / 10.0)
    
    rand_mask = np.random.rand(nbs, total_t) < (1.0 / M)
    high_inter_noise = (M * all_inter_linear) + noise_floor
    low_inter_noise = (M * all_inter_linear * (10**(-1.5))) + noise_floor
    
    inter_noise = np.where(rand_mask, high_inter_noise, low_inter_noise)

    # M * ps_linear mirrors the M-fold scaling on the interference term.
    # See the matching fix in MCSEvaluation / CheckHO_Failure above.
    return 10 * np.log10(M * ps_linear + 1e-15) - 10 * np.log10(inter_noise + 1e-15)

def vectorized_oracle(channels: np.ndarray, system_params: dict) -> tuple:
    input_ndim = channels.ndim
    if input_ndim == 1:
        channels_2d = channels.reshape(-1, 1)
    else:
        channels_2d = channels

    snir = calculate_snir_matrix(channels_2d, system_params)
    
    idx = np.searchsorted(system_params.get("SINRThreshold"), snir, side='right') - 1
    idx = np.clip(idx, 0, len(system_params.get("SpectralEff")) - 1)
    mcs = system_params.get("SpectralEff")[idx]
    
    if input_ndim == 1:
        return mcs.flatten(), snir.flatten()
    return mcs, snir

def vectorized_hof(channels: np.ndarray, system_params: dict) -> np.ndarray:
    input_ndim = channels.ndim
    if input_ndim == 1:
        channels_2d = channels.reshape(-1, 1)
    else:
        channels_2d = channels

    bler = np.array([
        1, 0.2617, 0.2370, 0.2103, 0.1828, 0.1558, 0.1302, 0.1067, 0.0859,
        0.0678, 0.0526, 0.0401, 0.0301, 0.0221, 0.0160, 0.0114, 0.0080,
        0.0056, 0.0038, 0.0025, 0.0017, 0.0011, 0.0007, 0.0004, 0.0003,
        0.0002, 0.0001
    ])

    snr_level = np.array([
        -np.inf, -1.7609, -1.6609, -1.5609, -1.4609, -1.3609, -1.2609,
        -1.1609, -1.0609, -0.9609, -0.8609, -0.7609, -0.6609, -0.5609,
        -0.4609, -0.3609, -0.2609, -0.1609, -0.0609, 0.0391, 0.1391,
        0.2391, 0.3391, 0.4391, 0.5391, 0.6391, 0.7391
    ])

    snir = calculate_snir_matrix(channels_2d, system_params)
    
    idx = np.searchsorted(snr_level, snir, side='right') - 1
    idx = np.clip(idx, 0, len(bler) - 1)
    pe = bler[idx]
    
    rand_vals = np.random.rand(channels_2d.shape[0], channels_2d.shape[1])
    hof = (rand_vals < pe).astype(np.int8)
    
    if input_ndim == 1:
        return hof.flatten()
    return hof