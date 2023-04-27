import numpy as np
import gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), is_slippery=True)

# Initialize the Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 10000

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Choose action: epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[int(state)])


        # Take the action
        next_state, reward, done, info = env.step(action)

        # Update the Q-table
        q_value = q_table[state, action]
        next_q_value = np.max(q_table[next_state])
        new_q_value = q_value + alpha * (reward + gamma * next_q_value - q_value)
        q_table[state, action] = new_q_value

        state = next_state

# Test the learned policy
num_test_episodes = 100
successes = 0
for episode in range(num_test_episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[char(state)])

        state, reward, done, info = env.step(action)

        if done:
            if reward == 1:
                successes += 1

success_rate = successes / num_test_episodes
print(f"Success rate: {success_rate}")
