# Future Improvements: RL Action Loop Architecture

## Strict 100ms Loop vs. Early Return RLF Tripwire

In the current implementation of the Reinforcement Learning environment (`ltm_gym.py`), we have adopted a **Strict 100ms Loop** for the agent's action resolution, while keeping the underlying physics simulation at a strict **10ms** tick rate (for parity with the legacy LTM implementation).

### The Adopted Approach: Strict 100ms Loop
When the RL agent (e.g., DQN) selects an action, the environment internally advances the 10ms physics generator exactly 10 times.
- **If an RLF occurs** (e.g., on tick 4), the environment applies a massive negative reward to the internal accumulator.
- For the remaining ticks (ticks 5-10), the environment forces a greedy fallback action to simulate a rapid reconnection attempt.
- The state is returned to the DQN *only* after the full 100ms has elapsed.
- **Pros:** The RL agent perceives the environment at perfectly spaced 100ms intervals. Moving averages (like MCS over 100ms) remain perfectly stable without temporal noise.
- **Cons:** The DQN does not see the exact state at the moment of the RLF; it only sees the state *after* the internal greedy recovery has finished the 100ms block.

### The Alternative: Early Return RLF Tripwire
An alternative approach discussed during the design phase was the **Early Return RLF Tripwire**.
In this architecture, if an RLF occurs at tick 4 (40ms):
- The environment halts the internal loop immediately.
- It applies the negative reward and returns the current RLF state to the DQN immediately, at $t + 40\text{ms}$.
- The DQN must then actively learn to select a recovery action.
- **Pros:** This aligns more naturally with the theoretical definition of a Markov Decision Process (MDP), which supports variable-time transitions (e.g., in robotic navigation, hitting a wall early instantly triggers a new state evaluation). The DQN sees the exact state that caused the failure.
- **Cons:** It introduces temporal variance into the observation space. A context vector that expects "100ms moving averages" will suddenly be fed data computed over only 40ms, which can inject noise and instability into the neural network's perception of velocity and signal trends.

### Conclusion
We chose the **Strict 100ms Loop** because maintaining the integrity of the 100ms moving averages was deemed more critical for stable learning than providing the DQN with the instantaneous failure state. However, the Early Return approach remains a viable alternative if future architectures remove fixed-time moving averages from the state space in favor of LSTMs or time-agnostic embeddings.
