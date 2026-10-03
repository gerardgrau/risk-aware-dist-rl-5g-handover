import subprocess
import os
import sys
import time

def run_ablation():
    quantiles = [10, 50, 100, 200]
    config_path = "configs/test-quantiles.yaml"
    device = "xpu" if "--cpu" not in sys.argv else "cpu"
    
    print(f"=== Starting Ablation Study on Quantiles: {quantiles} ===")
    print(f"Device: {device}")
    
    results_summary = []
    
    for n in quantiles:
        print(f"\n>>> Running QRDQN with N={n}...")
        temp_config = f"configs/temp_ablation_N{n}.yaml"
        
        with open(config_path, 'r') as f:
            lines = f.readlines()
            
        with open(temp_config, 'w') as f:
            for line in lines:
                if 'num_quantiles:' in line:
                    f.write(f"  num_quantiles: {n}\n")
                else:
                    f.write(line)
        
        cmd = [
            "venv-RL/bin/python3", "src/main.py",
            "--config", temp_config,
            "--description", f"ablation-quantiles-N{n}",
            "--device", device,
            "--agents", "qrdqn"
        ]
        
        start_time = time.time()
        subprocess.run(cmd)
        end_time = time.time()
        
        duration = end_time - start_time
        results_summary.append((n, duration))
        print(f"--- N={n} completed in {duration:.2f} seconds ---")
        
        os.remove(temp_config)
        
    print("\n" + "="*40)
    print("      ABLATION STUDY SUMMARY")
    print("="*40)
    print(f"{'Quantiles (N)':<15} | {'Duration (s)':<15}")
    print("-" * 35)
    for n, dur in results_summary:
        print(f"{n:<15} | {dur:<15.2f}")
    print("="*40)

if __name__ == "__main__":
    run_ablation()
