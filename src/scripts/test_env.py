import gymnasium as gym
import ale_py

# 1. Explicitly register the Arcade Learning Environment (Atari) games
gym.register_envs(ale_py)

# 2. Initialize the environment. 'human' mode opens a window so you can watch.
env = gym.make("ALE/Breakout-v5", render_mode="human")

# 3. Reset the environment to get the initial state
observation, info = env.reset()

for step in range(1000):
    # Sample a random action from the environment's action space
    action = env.action_space.sample()
    
    # Apply the action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # If the game ends or times out, reset it
    if terminated or truncated:
        print(f"Game ended after {step} steps. Resetting...")
        observation, info = env.reset()

# Close the rendering window when done
env.close()


