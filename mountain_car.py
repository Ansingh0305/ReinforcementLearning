import gym
import numpy as np
import pickle
import time  # Import time module for delays

# Load the Q-table
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Load the environment with render_mode="human"
env = gym.make("MountainCar-v0", render_mode="human")

# Discretize state space (use the same logic as in the training script)
num_bins = (20, 20)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(b[0], b[1], num_bins[i] - 1) for i, b in enumerate(state_bounds)]

def discretize_state(state):
    """Convert continuous state to discrete indices."""
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))

# Render the agent's behavior using the Q-table
state, _ = env.reset()
state = discretize_state(state)

done = False
total_reward = 0
terminated = 0
while not done:
    env.render()  # Display the environment
    time.sleep(0.05)  # Add a small delay between frames
    # Select the action with the highest Q-value
    action = np.argmax(q_table[state])
    # Take the action in the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = discretize_state(next_state)
    total_reward += reward

    if terminated:
        done = True

# Keep the render window open for a few seconds after completion
# time.sleep(5)  # Delay for 5 seconds to view the final state
env.close()
print(f"Total Reward: {total_reward}")
