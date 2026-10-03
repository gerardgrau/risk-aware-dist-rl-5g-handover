import os
import sys
import numpy as np

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.agents.standard.cmab import CMABAgent
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config
from src.distrl.utils.metrics import calculate_8_metrics

def main():
    print("Starting CMAB Integration Smoke Test...")
    Config.set_config_path("configs/cmab.yaml")
    config = Config.get()
    
    # Run test on 1 UE
    config["simulation"]["ue_number"] = 1
    
    env = LTMEnv(config=config)
    agent = CMABAgent(config["agent"], env.observation_space, env.action_space)
    
    # Seed matching
    np.random.seed(42 + 1)
    state, info = env.reset()
    agent.reset()
    agent.set_env(env)
    
    done = False
    episode_reward = 0.0
    steps = 0
    
    # Run in high-res callback mode
    while not done:
        state, r, done, _, info = env.step(
            0, high_res_callback=agent.select_action
        )
        episode_reward += float(r)
        steps += 1
        
    print("Episode completed successfully!")
    print(f"Steps: {steps}, Total Reward: {episode_reward:.4f}")
    
    # Calculate metrics
    m = calculate_8_metrics(
        mcs_history=info["metrics"]["mcs"],
        rlf_history=info["metrics"]["rlf"],
        ho_history=info["metrics"]["ho"],
        hof_history=info["metrics"]["hof"],
        pp_history=info["metrics"]["pp"],
        reserved_history=info["metrics"]["reserved"],
        config=config,
    )
    
    print("\n----- Episode Metrics -----")
    for k, v in m.items():
        print(f"{k:20}: {v:.4f}")
        
    env.close()
    print("Smoke Test Passed!")

if __name__ == "__main__":
    main()
