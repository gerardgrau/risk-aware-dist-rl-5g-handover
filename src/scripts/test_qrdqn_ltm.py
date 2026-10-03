import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.replay_buffer import ReplayBuffer

def train_qrdqn_ltm_minimal():
    env = LTMEnv()
    
    config = {
        "num_quantiles": 30,
        "kappa": 1.0,
        "hidden_dims": [64, 64],
        "lr": 1e-3,
        "gamma": 0.9,
        "tau": 0.01,
        "buffer_size": 5000,
        "batch_size": 64,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_mult": 0.99
    }
    
    device = "cpu"
    agent = QRDQNAgent(config, env.observation_space, env.action_space, device=device)
    buffer = ReplayBuffer(
        config["buffer_size"], env.observation_space.shape,
        action_dim=env.action_space.n,
    )
    
    print(f"Starting QRDQN training on LTM-HO simulation ({device})...")
    
    num_episodes = 20
    epsilon = config["epsilon_start"]
    
    pbar = tqdm(range(num_episodes), desc="QRDQN Test")
    for ep in pbar:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            mask = env.valid_action_mask()
            action = agent.select_action(state, epsilon, mask)
            next_state, reward, done, _, _ = env.step(action)

            buffer.push(state, action, reward, next_state, done,
                        env.valid_action_mask())
            state = next_state
            episode_reward += reward
            
            if len(buffer) > config["batch_size"]:
                agent.train_step(buffer.sample(config["batch_size"], device=device))
                
            # Decay epsilon per step
            epsilon = max(config["epsilon_end"], epsilon * config['epsilon_mult'])
            
        pbar.set_postfix({"reward": f"{episode_reward:.1f}"})

    print("QRDQN LTM-HO validation successful.")

if __name__ == "__main__":
    train_qrdqn_ltm_minimal()
