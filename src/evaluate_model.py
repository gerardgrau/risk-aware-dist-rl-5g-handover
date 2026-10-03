import torch
import numpy as np
import os
import sys
import pandas as pd
from typing import Any

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.utils.evaluation import run_evaluation

def evaluate(agent_type: str, model_path: str, config_path: str = "configs/config.yaml", output: str = None):
    print(f"\n--- Starting Standalone Evaluation ---")
    print(f"Agent:  {agent_type}")
    print(f"Model:  {model_path}")
    print(f"Config: {config_path}")
    
    # Load custom config
    Config.set_config_path(config_path)
    config = Config.get()
    
    # Ensure standalone evaluation uses the 1000 UEs test dataset
    import copy
    config = copy.deepcopy(config)
    test_dir = "data/ChannelGains"
    for candidate in ["data/test", "data/5g_advanced_ltm_channel_trajectories/test", "data/ChannelGains"]:
        if os.path.exists(candidate):
            test_dir = candidate
            break
    config['paths']['channel_data_directory'] = test_dir
    config['paths']['precomputed_data_directory'] = "data/Precomputed"
    config['simulation']['ue_number'] = 1000
    
    # Ensure we use the correct number of UEs
    
    # New Protocol: All users are used for both training and evaluation
    env = LTMEnv(config=config)
    
    if agent_type.lower() == "dqn":
        agent = DQNAgent(config['agent'], env.observation_space, env.action_space)
    elif agent_type.lower() == "qrdqn":
        agent = QRDQNAgent(config['agent'], env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    print("Loading weights...")
    agent.load(model_path)
    
    # Setup directory for results if output is provided
    experiment_dir = "."
    if output:
        experiment_dir = os.path.dirname(output) or "."
        os.makedirs(experiment_dir, exist_ok=True)
        
    # Use the unified evaluation utility
    run_evaluation(
        agent=agent,
        config=config,
        experiment_dir=experiment_dir,
        agent_type=agent_type,
        seed=42,
        save_results=True if output else False,
        output_prefix=output
    )

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="dqn or qrdqn")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, help="Path prefix to save evaluation summary and raw metrics CSVs")
    args = parser.parse_args()
    
    evaluate(args.agent, args.model, args.config, args.output)
