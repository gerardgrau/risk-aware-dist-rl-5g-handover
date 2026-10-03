import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import glob
from typing import Any

ChannelDirectory = "data/ChannelGains"

# Find all files
files = glob.glob(os.path.join(ChannelDirectory, "ChannelGainBSUE_User*.mat"))

UE_Number = len(files)
# UE_Number = 5  # To test, limitar a 5 UEs

print(f"Detected {len(files)} UE channel files. Simulating {UE_Number} UEs.")

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

ReceiverSensitivity = -95

# System parameters
System = {
    "TxPower": 25,  # dBm
    # The paper uses 25
    "NoiseLevel": -174 + 10 * np.log10(20*1e6),  # dBm
    # Added the 20 MHz bandwidth
    
    "SINRThreshold": np.array([
        -np.inf, -3, -2, 0, 2, 4, 6, 7, 10, 12, 14, 16, 20, 
        22, 24, 26, 28, 30, 32, 35, 38, 40, 42, 44, 46, 48
    ]),
    "SpectralEff": np.array([
        0, 0.24, 0.38, 0.60, 0.88, 1.18, 1.46, 1.70, 1.92, 
        2.40, 2.92, 3.40, 3.60, 4.14, 4.74, 5.28, 5.58, 5.7, 
        5.85, 5.92, 6.64, 7.12, 7.44, 7.50, 8.30, 9.30
    ])
    # The valid entries are the first ones

    # "SINRThreshold": np.array([
    #     -np.inf, -6.5, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.5, 10.5, 
    #     12.5, 14.5, 16.5, 19.0, 21.5, 24.0
    # ]),
    # "SpectralEff": np.array([
    #     0, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 
    #     2.73, 3.32, 3.90, 4.52, 5.12, 5.55
    # ])
    # These last entries are NOT the valid ones
}

# Time parameters
Time = {
    "TotalSimTime": 60,         # 60 segundos
    "TimeStep": 10e-3           # 1 ms
}

# BS parameters
BS = {"Number": 7}  # set to 7 for the hexagonal-cell layout
NBS = BS["Number"] * 3  # 3 sectors per BS
HO = {}

# HO["Prep"] = {
#     "PeriodicityRSRPMeasurement": 20e-3,     # 20 ms
#     "AverageRSRPMeasument_NL1": 10,          # FIR filter length
#     "kL3": 8,                                 # L3 IIR constant
#     "alphaIIRfilter": 2 ** (-8 / 4),          # 2^(-kL3/4)
#     "PreparationPowerOffset": -3,             # dB
#     "PreparationTime": 40e-3,                 # 40 ms
#     "ExecPowerOffset": 3,                     # dB
#     "MaxNumberPreparedBS": 4                  # Table II: predicted top-4 neighbors
# }

HO["Prep"] = {
    "PeriodicityRSRPMeasurement": 20e-3,      # 20 ms
    "AverageRSRPMeasument_NL1": 10,           # FIR filter length
    "kL3": 8,                                 # L3 IIR constant
    "PreparationPowerOffset": -3,             # dB
    "PreparationTime": 40e-3,                 # 40 ms
    "ExecPowerOffset": 3,                     # dB
    "MaxNumberPreparedBS": 5                  # reference value (see CLAUDE.md parity notes)
}
HO["Prep"]["alphaIIRfilter"] = 2 ** (-HO["Prep"]["kL3"] / 4)      # 2^(-kL3/4)
# Confirmed against the reference (the constants below too)

# Time variables
Time_PingPong = 1 # Ping-pong event: HO is successful, but UE HO to previous cell within 1 second
Time_InitialSync_0 = 0.0115
Time_MeasReportL3_1 = 0.01
Time_RRCTransfer2 = 0.01
Time_RRCConf3 = 0.022
Time_RRCReconf4_5 = 0.02
Time_MeasReportL1_67 = 0.01
Time_HOdecision_8 = 0.01
Time_LLHOCommand_9 = 0.02
Time_RA_10 = 10e-3                     # Time spent in RACH
Time_ContextRelease_11 = 20e-3         # Time for release and switching: between 10 and 50ms

# ============================================================
# FUNCIONES PRINCIPALES
# ============================================================
def get_realistic_interference(ch_vector, serving_idx, all_inter_linear, system_params):
    """
    Implements ICIC (Inter-Cell Interference Coordination).
    ch_vector: All BS gains for the current UE at time t
    serving_idx: Current BS
    all_inter_linear: Raw total interference
    """
    raise NotImplementedError()
    noise_floor = 10**(system_params["NoiseLevel"] / 10.0)
    
    if serving_idx == -1:
        return all_inter_linear + noise_floor, False

    # 1. Identify the strongest neighbor to check HO proximity
    neighbor_indices = np.delete(np.arange(len(ch_vector)), serving_idx)
    target_idx = neighbor_indices[np.argmax(ch_vector[neighbor_indices])]
    
    # 2. Calculate the Handover Margin (dB)
    # How close is the neighbor to taking over?
    serving_pwr = ch_vector[serving_idx]
    target_pwr = ch_vector[target_idx]
    ho_margin_db = 10 * np.log10(serving_pwr / (target_pwr + 1e-15))
    
    # 3. ICIC Trigger Logic
    # In 5G, if the neighbor is within 3-6dB, the network coordinates.
    icic_active = ho_margin_db < 7.0  
    
    if icic_active:
        # ICIC reduction (0.1 = -10dB). 
        # This is the "Safety Window" for the handover to succeed.
        reduction_factor = 0.1
        signal = serving_pwr + 3  # Assume the network boosts the signal by 3 dB during HO prep
        inter_noise = (all_inter_linear * reduction_factor) + noise_floor
    else:
        # Standard Reuse 1 - No protection
        inter_noise = all_inter_linear + noise_floor
        signal = serving_pwr
        
    return inter_noise, icic_active

def MCSEvaluation(serving_sector, channels, System, Sync):
    RLF = 0
    serving_channel = channels[serving_sector]
    reuse_factor = 3
    all_sectors = np.arange(len(channels)) 
    target_mask = (all_sectors % reuse_factor) == (serving_sector % reuse_factor)
    Ps = 10 ** ((serving_channel + System["TxPower"]) / 10)
    relevant_channels = channels[target_mask]
    Inter = (relevant_channels + System["TxPower"]) / 10.0
    # print(f"Serving channel (dB): {serving_channel:.2f}, Power (linear): {Ps:.2e}")

    # Sum of interference in linear scale (10^(P/10))
    AllInter = np.sum(10**Inter) - Ps
    # AllInter = AllInter * 0.1  # Attenuation factor to simulate better conditions during HO
    M = 3
    noise_linear = 10**(System["NoiseLevel"] / 10.0)

    if np.random.rand() < (1 / M):
        # Case: High interference scenario
        Inter_Noise = M * AllInter + noise_linear
    else:
        # Case: Low interference scenario (attenuated by 15 dB)
        Inter_Noise = M * AllInter * 10**(-1.5) + noise_linear
    # Temporary debug line
    # Inter_Noise = 10**(System["NoiseLevel"]/10)

    # Inter_Noise, icic_on = get_realistic_interference(channels, serving_sector, AllInter, System)
    # print(f"Serving channel (dB): {serving_channel:.2f}, Interference+Noise (dB): {10*np.log10(Inter_Noise):.2f}, ICIC active: {icic_on}")
    SNIR = 10 * np.log10(M * Ps) - 10 * np.log10(Inter_Noise)
    idx = np.where(System["SINRThreshold"] <= SNIR)[0]
    MCS = System["SpectralEff"][idx[-1]] if len(idx) > 0 else 0
    # if MCS == 0:
    #     print(f"MCS: {MCS:.2f}, SNIR: {SNIR:.2f} dB, ICIC: {'ON' if icic_on else 'OFF'}, idx={idx}")

    # print(f"SNIR (dB): {SNIR:.2f}, MCS: {MCS:.2f}")
    # Synchronization
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
    BLER = np.array([
        1, 0.2617, 0.2370, 0.2103, 0.1828, 0.1558, 0.1302, 0.1067, 0.0859,
        0.0678, 0.0526, 0.0401, 0.0301, 0.0221, 0.0160, 0.0114, 0.0080,
        0.0056, 0.0038, 0.0025, 0.0017, 0.0011, 0.0007, 0.0004, 0.0003,
        0.0002, 0.0001
    ])

    SNR_level = np.array([
        -np.inf, -1.7609, -1.6609, -1.5609, -1.4609, -1.3609, -1.2609,
        -1.1609, -1.0609, -0.9609, -0.8609, -0.7609, -0.6609, -0.5609,
        -0.4609, -0.3609, -0.2609, -0.1609, -0.0609, 0.0391, 0.1391,
        0.2391, 0.3391, 0.4391, 0.5391, 0.6391, 0.7391
    ])

    serving_channel = channels[serving_sector]
    reuse_factor = 3
    all_sectors = np.arange(len(channels)) 
    target_mask = (all_sectors % reuse_factor) == (serving_sector % reuse_factor)
    Ps = 10 ** ((serving_channel + System["TxPower"]) / 10)

    relevant_channels = channels[target_mask]
    Inter = (relevant_channels + System["TxPower"]) / 10.0
    AllInter = np.sum(10**Inter) - Ps
    # AllInter = AllInter * 0.1  # Attenuation factor to simulate better conditions during HO
    M = 3
    noise_linear = 10**(System["NoiseLevel"] / 10.0)

    if np.random.rand() < (1 / M):
        # Case: High interference scenario
        Inter_Noise = M * AllInter + noise_linear
    else:
        # Case: Low interference scenario (attenuated by 15 dB)
        Inter_Noise = M * AllInter * 10**(-1.5) + noise_linear
    # Temporary debug line
    # Inter_Noise = 10**(System["NoiseLevel"]/10)

    # Inter_Noise, icic_on = get_realistic_interference(channels, serving_sector, AllInter, System)

    # print(f"Serving channel (dB): {serving_channel:.2f}, Interference+Noise (dB): {10*np.log10(Inter_Noise):.2f}, ICIC active: {icic_on}")
    SNIR = 10 * np.log10(M * Ps) - 10 * np.log10(Inter_Noise)
    idx = np.where(SNR_level <= SNIR)[0]
    Pe = BLER[idx[-1]]

    return np.random.rand() < Pe

def PerformanceEvaluation(MCS, ServingCell_channel, ping_pong, ReservedBSSectors, HOF, HO_event, RLF, Time):
    Minutes = MCS.shape[0] * Time["TimeStep"] / 60

    Performance = {}
    Performance["Capacity"] = MCS
    Performance["RL_problems"] = np.sum(RLF) / Minutes
    Performance["Number_HO"] = np.sum(HO_event) / Minutes
    Performance["Number_ping_pongs"] = np.sum(ping_pong) / Minutes
    Performance["Reliability"] = 100 - np.mean(MCS == 0) * 100
    # FIX: count only 0->1 entering transitions (the MATLAB simulator uses > 0).
    # The previous >= 0 counted every non-decreasing tick (~126k/min); > 0 gives
    # the true discrete preparation rate (~26/min).
    Performance["Number_cell_preparations"] = np.sum((ReservedBSSectors[:, 1:] - ReservedBSSectors[:, :-1]) > 0) / Minutes
    Performance["Resource_reservation"] = np.sum(ReservedBSSectors) / ReservedBSSectors.size * 100
    Performance["HOF"] = np.sum(HOF) / Minutes

    return Performance


# ============================================================
# MAIN
# ============================================================

def run_simulation():
        

    Performance_all = []
    Metrics = []

    for indUE in range(0, UE_Number):
        print(f"Simulando UE {indUE+1}/{UE_Number}...")
        # Per-UE seed for reproducibility (matches the seed set externally in
        # src/scripts/verify_simulation_parity.py and other gym wrappers).
        np.random.seed(42 + indUE + 1)
        filename = os.path.join(ChannelDirectory, f"ChannelGainBSUE_User{indUE+1}.mat")
        mat_data = loadmat(filename)
        # Channel = mat_data['ChannelBS2UE'] # shape = (T, BS, sectors)
        Channel = mat_data['ChannelBS2UE_noRIS'] # shape = (T, BS, sectors) # the reference uses this channel for its figures

        NBS = BS["Number"] * 3
        ChBS2UE = np.zeros((NBS, Channel.shape[0]))

        idx = 0
        TotalTime = Channel.shape[0]
        Max_iter = TotalTime
        for b in range(Channel.shape[1]):
            for s in range(Channel.shape[2]):
                # print(f"Processing UE {indUE}, BS {b}, Sector {s}...")
                # print(f"idx={idx}")
                ch = Channel[:, b, s]
                ChBS2UE[idx, :] = ch
                idx += 1

        # Filtros L1 y L3
        M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
        b = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]

        L1 = lfilter(b, 1, ChBS2UE[:, ::M], axis=1)
        L3 = lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1)

        PL1 = np.repeat(L1, M, axis=1)[:, :ChBS2UE.shape[1]] # PL1 = RSRP filtrat
        PL3 = np.repeat(L3, M, axis=1)[:, :ChBS2UE.shape[1]] # PL3 = PL1 filtrat

        # Metric initialization
        # ServingBSSector = np.zeros(Max_iter, dtype=int)
        # Initialize an array of length Max_iter filled with -1
        ServingBSSector = np.full(Max_iter, -1, dtype=int)
        MCS = np.zeros(Max_iter)
        HOF = np.zeros(Max_iter)
        RLF = np.zeros(Max_iter)
        HO_event = np.zeros(Max_iter)
        ping_pong = np.zeros(Max_iter)
        ReservedBSSectors = np.zeros((NBS, Max_iter))

        Sync = {
            "N310": 4,
            "N311": 2,
            "T310": 50,     # 50 time steps of 10 ms = 500 ms
            "out_sync_count": 0,
            "in_sync_count": 0,
            "t310_running": False,
            "t310_counter": np.inf
        }

        ListBSPrepared = np.zeros(NBS, dtype=int)
        TimerEntering = np.zeros(NBS)
        TimerLeaving = np.zeros(NBS)

        t = 0
        NextBSSector = -1
        former_BS = -1
        former_HO_time = -np.inf

        while t < (Max_iter - 10):
            # FIX: Save preparation state BEFORE time increments to avoid the missing prep hole bug
            ReservedBSSectors[:, t] = ListBSPrepared
            
            # print(f"sample={t}, ServingBSSector={ServingBSSector[t]}")
            if ServingBSSector[t] >= 0 and not RLF[t]:
                # cell search

                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                if RLF[t]:
                    NextBSSector = -1

            t += 1

            while NextBSSector == -1:

                # GREEDY SELECTION: Pick the strongest Base Station.
                # Note: In the RL simulation (ltm_gym.py), this hardcoded greedy 
                # logic is bypassed to allow the agent to make the decision.

                Pbest = np.max(ChBS2UE[:, t])
                Best = np.argmax(ChBS2UE[:, t])

                if Pbest + System["TxPower"] > ReceiverSensitivity:
                    NextBSSector = Best
                    MCS[t], RLF[t], Sync = MCSEvaluation(NextBSSector, ChBS2UE[:, t], System, Sync)
                    if MCS[t] == 0:
                        NextBSSector = -1
                    else:
                        # Reset synchronization counters on cell change
                        Sync["out_sync_count"] = 0
                        Sync["in_sync_count"] = 0
                        Sync["t310_running"] = False
                        Sync["t310_counter"] = np.inf
                t += 1


            # 5G HO logic: preparation, execution, completion (Python version)

            # Serving sector at current time
            ServingBSSector[t] = NextBSSector

            # Metrics
            ReservedBSSectors[:, t] = ListBSPrepared
            MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
            if RLF[t]:
                NextBSSector = -1
                continue

            # ============================
            # HO PREPARATION
            # ============================

            # 1. L3 Measurement report
            PL3_report = PL3[:, t]

            Tf = min(t + int(np.ceil(Time_MeasReportL3_1 / Time["TimeStep"])), Max_iter-1)
            while t < Tf:
                ReservedBSSectors[:, t] = ListBSPrepared
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                if RLF[t]:
                    break
                t += 1
                ServingBSSector[t] = ServingBSSector[t - 1]

            if RLF[t]:
                NextBSSector = -1
                continue

            # 2 & 3. UL RRC transfer & preparation of target cells and RRC configuration
            Tf = min(t + int(np.ceil((Time_RRCTransfer2 + Time_RRCConf3) / Time["TimeStep"])), Max_iter-1)
            while t < Tf:
                ReservedBSSectors[:, t] = ListBSPrepared
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                if RLF[t]:
                    break
                t += 1
                ServingBSSector[t] = ServingBSSector[t - 1]

            if RLF[t]:
                NextBSSector = -1
                continue

            # Cell preparation entering condition
            In_condition = PL3_report > (PL3_report[ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
            TimerEntering = (TimerEntering + Time["TimeStep"]) * In_condition
            ListBSPrepared = np.logical_or(ListBSPrepared, (TimerEntering > HO["Prep"]["PreparationTime"]))

            # Cell preparation leaving condition
            Out_condition = PL3_report < (PL3_report[ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
            TimerLeaving = (TimerLeaving + Time["TimeStep"]) * Out_condition
            ListBSPrepared = np.logical_and(ListBSPrepared, ~(TimerLeaving > HO["Prep"]["PreparationTime"]))

            # Reduce list if too long
            if np.sum(ListBSPrepared) > HO["Prep"]["MaxNumberPreparedBS"]:
                # Effective power only on the prepared cells
                metric = (10 ** (PL3_report / 10)) * ListBSPrepared
                I_sorted = np.argsort(metric)[::-1]
                # Zero out the weakest ones
                ListBSPrepared[I_sorted[HO["Prep"]["MaxNumberPreparedBS"]:]] = 0

            # 4 & 5. RRC configuration
            Tf = min(t + int(np.ceil(Time_RRCReconf4_5 / Time["TimeStep"])), Max_iter-1)
            while t < Tf:
                ReservedBSSectors[:, t] = ListBSPrepared
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                if RLF[t]:
                    break
                t += 1
                ServingBSSector[t] = ServingBSSector[t - 1]

            if RLF[t]:
                NextBSSector = -1
                continue

            # ============================
            # HO EXECUTION & COMPLETION
            # ============================

            # 6 & 7 & 8. L1 measurement report and cell change decision
            PL1_report = PL1[:, t]

            HO_condition = np.logical_and(
                ListBSPrepared, # note: not pre-filtering by prepared has no effect here
                PL1_report > (PL1_report[ServingBSSector[t]] + HO["Prep"]["ExecPowerOffset"])
            )

            Tf = min(t + int(np.ceil((Time_MeasReportL1_67 + Time_HOdecision_8) / Time["TimeStep"])), Max_iter-1)
            while t < Tf:
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                ReservedBSSectors[:, t] = ListBSPrepared
                if RLF[t]:
                    break
                t += 1
                ServingBSSector[t] = ServingBSSector[t - 1]

            if RLF[t]:
                NextBSSector = -1
                continue

            if np.sum(HO_condition) > 0:
                # Positive decision for HO
                t0 = t
                HO_event[t0] = 1

                # GREEDY SELECTION: Choose the best candidate cell based on signal strength.

                metric = (10 ** (PL1_report / 10)) * HO_condition
                I = np.argmax(metric)

                # 9. Lower layer command
                Tf = min(t + int(np.ceil(Time_LLHOCommand_9 / Time["TimeStep"])), Max_iter-1)
                while t < Tf:
                    ReservedBSSectors[:, t] = ListBSPrepared
                    MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                    if RLF[t]:
                        break
                    t += 1
                    ServingBSSector[t] = ServingBSSector[t - 1]

                if RLF[t]:
                    NextBSSector = -1
                    continue

                # 10. Random access
                t = min(t + int(np.ceil(Time_RA_10 / Time["TimeStep"])), Max_iter-1)
                HOF[t] = CheckHO_Failure(I, ChBS2UE[:, t], System)

                if HOF[t] == 0:
                    # 11. UE context release and path switch
                    NextBSSector = I
                    t = min(t + int(np.ceil(Time_ContextRelease_11 / Time["TimeStep"])), Max_iter-1)

                    # Recursos reservados durante el HO
                    ReservedBSSectors[:, t0:t + 1] = np.tile(ListBSPrepared.reshape(-1, 1), (1, t - t0 + 1))
                    ListBSPrepared[NextBSSector] = 0
                else:
                    NextBSSector = -1  # HOF: connection lost

                # Ping-pong
                ping_pong[t0] = (NextBSSector == former_BS) and ((t0 - former_HO_time) < int(np.floor(Time_PingPong / Time["TimeStep"])))
                former_BS = ServingBSSector[t0]
                former_HO_time = t0

        # Final evaluation
        Performance = PerformanceEvaluation(MCS, ServingBSSector, ping_pong, ReservedBSSectors, HOF, HO_event, RLF, Time)
        Performance_all.append(Performance)

        Metrics.append({
            "MCS": MCS,
            "ping_pong": ping_pong,
            "HOF": HOF,
            "HO_event": HO_event,
            "RLF": RLF,
            "ServingBSSector": ServingBSSector
        })

    return Performance_all, Metrics


# ============================================================
# EXECUTION
# ============================================================

def network_loader(net_file):
    # Would load the network topology, BS positions, etc.
    import sumolib.net as net
    net = net.readNet(net_file)
    G = nx.DiGraph()

    for edge in net.getEdges():
        from_node = edge.getFromNode()
        to_node = edge.getToNode()

        # Coordenadas X/Y de SUMO
        x1, y1 = from_node.getCoord()
        x2, y2 = to_node.getCoord()

        # Add nodes with position
        G.add_node(from_node.getID(), pos=(x1, y1))
        G.add_node(to_node.getID(), pos=(x2, y2))

        # Add edge with length
        G.add_edge(from_node.getID(), to_node.getID(), length=edge.getLength())

    min_x, min_y, max_x, max_y = net.getBoundary()  # also returns (min_x, min_y, max_x, max_y)
    pos = nx.get_node_attributes(G, "pos")
    off_x, off_y = net.getLocationOffset()
    return G, pos, (min_x, min_y, max_x, max_y), off_x, off_y
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import networkx as nx    
    import pandas as pd
    Performance_all, Metrics = run_simulation()
    print("Simulation complete.")
    
    # Minimal check for optional SUMO files
    net_file = "data/SUMO_Network/osm.net.xml"
    if os.path.exists(net_file):
        G, pos, bounds, off_x, off_y = network_loader(net_file)
    
    bs_file = "data/sumo_5g_base_stations.csv"
    if os.path.exists(bs_file):
        bs_df = pd.read_csv(bs_file)
        
    traj_file = "data/SUMO_Network/fcd.pkl"
    if os.path.exists(traj_file):
        df = pd.read_pickle(traj_file)
        veh_list = df["vehicle"].unique()
        
    # Print summaries
    print("\n===== PERFORMANCE SUMMARY =====")
    for ue_id, perf in enumerate(Performance_all):
        print(f"\nUE {ue_id+1} ({len(perf['Capacity']) * Time['TimeStep'] / 60:.3f} minutes):")
        print(f"  Capacity (avg MCS): {np.mean(perf['Capacity']):.2f}")

        for key, value in perf.items():
            if key == "Capacity": continue
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: array of length {len(value)}")

        # fig, ax = plt.subplots(figsize=(12, 6))
        # # 1. Draw the Road Network (The Background)
        # nx.draw(G, pos, ax=ax, node_size=0, edge_color="lightgray", width=1, alpha=1, label="Road Network")
        
        # # 2. Draw the Base Stations
        # ax.scatter(bs_df['x'], bs_df['y'], c="red", s=30, zorder=10, label="5G BS")

        # # 3. Draw the UE Trajectory
        # v_df = df[df["vehicle"] == veh_list[ue_id]].sort_values("time")
        # avg_speed = v_df["speed"].mean()
        # max_speed = v_df["speed"].max()
        # std_speed = v_df["speed"].std()

        # print(f"Average Speed: {avg_speed:.2f} m/s ({avg_speed * 3.6:.2f} km/h)")
        # print(f"Max Speed:     {max_speed:.2f} m/s ({max_speed * 3.6:.2f} km/h)")
        # print(f"Speed Variability (Std Dev): {std_speed:.2f} m/s")
        # ax.plot(v_df["x_real"].values+off_x, v_df["y_real"].values+off_y, c="blue", linewidth=2, label="UE Trajectory")
        # plt.show()
    # print("\n===== METRICS SUMMARY =====")
    # for ue_id, met in enumerate(Metrics):
    #     print(f"\nUE {ue_id}:")
    #     for key, value in met.items():
    #         print(f"  {key}: array of length {len(value)}")

    import pickle

    save_dir = "results/simulations"
    os.makedirs(save_dir, exist_ok=True)
    # Save Performance_all 
    with open(os.path.join(save_dir, "Performance_all.pkl"), "wb") as f: pickle.dump(Performance_all, f)
    # Save Metrics
    with open(os.path.join(save_dir, "Metrics.pkl"), "wb") as f: pickle.dump(Metrics, f)
    print("Results saved in folder:", save_dir)

    # To load
    # with open("Performance_all.pkl", "rb") as f:
    # Performance_all = pickle.load(f)

    # with open("Metrics.pkl", "rb") as f:
    #     Metrics = pickle.load(f)
