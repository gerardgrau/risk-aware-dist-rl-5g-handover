import torch
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.replay_buffer import ReplayBuffer

def run_device_benchmark(device_name: str, num_episodes: int = 10):
    print(f"\n>>> Benchmarking Device: {device_name.upper()}")
    config = Config.get()
    agent_config = config['agent']
    
    # Use simulation time from config to reflect production workload
    env = LTMEnv(config=config)
    agent = QRDQNAgent(agent_config, env.observation_space, env.action_space, device=device_name)
    buffer = ReplayBuffer(
        agent_config['buffer_size'], env.observation_space.shape,
        action_dim=env.action_space.n,
    )
    
    batch_size = agent_config.get('batch_size', 64)
    epsilon = 0.1 # Fixed low epsilon for benchmarking
    
    start_time = time.time()
    total_steps = 0
    
    pbar = tqdm(range(num_episodes), desc=f"Benchmarking {device_name.upper()}")
    for ep in pbar:
        state, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            
            # Train EVERY step once buffer is ready (Logic identical to src/main.py)
            if len(buffer) > batch_size:
                agent.train_step(buffer.sample(batch_size, device=device_name))
        
        elapsed = time.time() - start_time
        pbar.set_postfix({"steps": total_steps, "elapsed": f"{elapsed:.1f}s"})

    end_time = time.time()
    total_time = end_time - start_time
    steps_per_sec = total_steps / total_time
    
    print(f"DONE: {device_name.upper()} | Total Time: {total_time:.2f}s | Steps/sec: {steps_per_sec:.2f}")
    env.close()
    
    return total_time, steps_per_sec

def main():
    devices_to_test = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices_to_test.append("xpu")
    else:
        print("Intel XPU not detected or not available. Only CPU will be benchmarked.")

    results = {}
    num_episodes = 60 # Definitive ~1h total run across CPU/XPU with QRDQN
    
    for d in devices_to_test:
        total_time, steps_per_sec = run_device_benchmark(d, num_episodes=num_episodes)
        results[d] = {
            "total_time": total_time,
            "steps_per_sec": steps_per_sec
        }

    # Summary Table
    print("\n" + "="*40)
    print(f"{'Device':<10} | {'Total Time (s)':<15} | {'Steps/sec':<10}")
    print("-"*40)
    for d, metrics in results.items():
        print(f"{d.upper():<10} | {metrics['total_time']:<15.2f} | {metrics['steps_per_sec']:<10.2f}")
    print("="*40)

    # Visualization
    if len(results) > 1:
        labels = [d.upper() for d in results.keys()]
        times = [m['total_time'] for m in results.values()]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, times, color=['skyblue', 'orange'])
        plt.ylabel('Total Time (seconds) - Lower is Better')
        plt.title(f'Device Performance Comparison ({num_episodes} episodes)')
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}s", ha='center', va='bottom')
        
        output_dir = "results/benchmarks"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "device_comparison.png")
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")

if __name__ == "__main__":
    main()
