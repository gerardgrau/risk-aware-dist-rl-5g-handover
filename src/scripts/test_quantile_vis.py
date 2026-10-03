import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.viz.plot import plot_quantiles

def visualize_distributional_risk():
    print("=== QRDQN Distributional Visualization ===")
    
    # 1. Setup Env and Load Config
    env = LTMEnv()
    config = Config.get()
    agent_cfg = config['agent']
    
    # 2. Setup Agent and Load Weights
    device = "cpu" # Use CPU for plotting
    agent = QRDQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    
    model_path = "results/models/qrdqn_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run src/main.py with qrdqn first.")
        return
        
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # 3. Sample a State
    # Reset until we find a state with some signal variability
    state, _ = env.reset()
    for _ in range(100): # Step a bit into the simulation
        action = agent.select_action(state, epsilon=0)
        state, _, done, _, _ = env.step(action)
        if done: state, _ = env.reset()
        
        # Check if we have multiple BS with decent signal
        if np.sum(state > -110) >= 2:
            break

    print(f"Sampling state at t={env.t}. RSRP values: {state}")
    
    # 4. Plot Quantiles
    save_dir = "results/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "quantile_dist_sample.png")
    
    # Only label the top 3 actions to keep the plot clean
    q_means = agent.q_net(torch.FloatTensor(state).unsqueeze(0)).mean(dim=2).detach().numpy()[0]
    best_actions = np.argsort(q_means)[-3:]
    action_names = [f"BS {i}" if i in best_actions else "__skip__" for i in range(env.action_space.n)]
    
    # Use a modified version of action_names that filtering skips
    filtered_names = [name for name in action_names if name != "__skip__"]
    
    # We need to modify plot_quantiles or handle filtering here
    # Let's just plot all but label the important ones
    plot_quantiles(agent, state, action_names=[f"BS {i}" for i in range(env.action_space.n)], save_path=save_path)
    
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_distributional_risk()
