import os
import glob
import re
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from tqdm import tqdm
import sys

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.physics import (
    System, Time, HO, NBS, physics_hash, vectorized_oracle, vectorized_hof
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def preprocess(data_dir=None, output_base=None):
    if data_dir is None:
        data_dir = "data/ChannelGains"
    if output_base is None:
        output_base = "data/Precomputed"
    os.makedirs(output_base, exist_ok=True)
    
    all_files = sorted(glob.glob(os.path.join(data_dir, "ChannelGainBSUE_User*.mat")), key=natural_sort_key)
    print(f"Found {len(all_files)} files to preprocess.")
    if not all_files:
        print("No files found to preprocess!")
        return
    
    # Auto-detect keys from the first file
    first_mat = loadmat(all_files[0])
    if "ChannelBS2UE" in first_mat:
        modes = {
            "no_ris": "ChannelBS2UE"
        }
    else:
        modes = {
            "no_ris": "ChannelBS2UE_noRIS",
            "with_ris": "ChannelBS2UE_RIS"
        }
    
    # ENSURE SYSTEM POWER IS 25 (Pure Paper)
    print(f"Using System TxPower: {System['TxPower']} dBm, Noise: {System['NoiseLevel']} dBm")
    cache_hash = physics_hash()
    print(f"Physics hash: {cache_hash}")


    for mode_name, mat_key in modes.items():
        mode_dir = os.path.join(output_base, mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        print(f"\n>>> Preprocessing mode: {mode_name} (using {mat_key})")
        
        for filename in tqdm(all_files, desc=f"Mode {mode_name}"):
            user_id = os.path.basename(filename).split('_')[-1].split('.')[0]
            out_filename = os.path.join(mode_dir, f"{user_id}_precomputed.npz")
            
            mat_data = loadmat(filename)
            raw_channel = mat_data[mat_key] 
            
            total_time = raw_channel.shape[0]
            ch_bs2ue = np.zeros((NBS, total_time), dtype=np.float32)
            idx = 0
            for b in range(raw_channel.shape[1]):
                for s in range(raw_channel.shape[2]):
                    ch_bs2ue[idx, :] = raw_channel[:, b, s]
                    idx += 1
            
            # Reproducible Stochasticity for Parity Audit
            numeric_id = int(re.search(r'\d+', user_id).group())
            np.random.seed(42 + numeric_id)
            
            # Positions
            ue_pos_complex = mat_data['UE'][0, 0]['Position'][0]
            ue_positions = np.stack([ue_pos_complex.real, ue_pos_complex.imag], axis=1).astype(np.float32)
            
            # Physics calculations
            all_mcs, all_snir = vectorized_oracle(ch_bs2ue, System)
            all_pe = vectorized_hof(ch_bs2ue, System)
            
            # Filtering (L1 and L3 RSRP)
            M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
            b_filt = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]
            L1 = lfilter(b_filt, 1, ch_bs2ue[:, ::M], axis=1)
            pl1 = np.repeat(L1, M, axis=1)[:, :total_time].astype(np.float32)
            pl3 = np.repeat(lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :total_time].astype(np.float32)
            
            # Save as compressed NumPy binary
            np.savez_compressed(
                out_filename,
                physics_hash=cache_hash,
                total_time=total_time,
                ch_bs2ue=ch_bs2ue,
                all_mcs_episode=all_mcs.astype(np.float32),
                all_snir_episode=all_snir.astype(np.float32),
                all_pe_episode=all_pe.astype(np.float32),
                ue_positions=ue_positions,
                pl1=pl1,
                pl3=pl3
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Input directory containing raw .mat files")
    parser.add_argument("--output_base", type=str, default=None,
                        help="Output directory to save precomputed .npz files")
    args = parser.parse_args()
    if args.data_dir is not None:
        preprocess(args.data_dir, args.output_base or "data/Precomputed")
    else:
        processed_any = False
        train_path = None
        for candidate in ["data/train", "data/5g_advanced_ltm_channel_trajectories/train", "data/ChannelGains_v2"]:
            if os.path.exists(candidate):
                train_path = candidate
                break
        if train_path:
            print(f">>> Auto-detected {train_path} (Training split)...")
            preprocess(train_path, "data/Precomputed_v2")
            processed_any = True

        test_path = None
        for candidate in ["data/test", "data/5g_advanced_ltm_channel_trajectories/test", "data/ChannelGains"]:
            if os.path.exists(candidate):
                test_path = candidate
                break
        if test_path:
            print(f">>> Auto-detected {test_path} (Test split)...")
            preprocess(test_path, "data/Precomputed")
            processed_any = True

        if not processed_any:
            preprocess("data/ChannelGains", "data/Precomputed")

