import gymnasium as gym
import numpy as np
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

map_size = 4
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size), is_slippery=True)

q_table = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001  # exponential decay rate for exploration probability

num_episodes = 50000

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state, :])  # exploit

        next_state, reward, done, _, info = env.step(action)
        next_state = int(next_state)  # Updated line

        # Update Q-table
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])

        state = int(env.reset())


    # Decay exploration rate
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

num_test_episodes = 5

for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    print(f"Episode {episode + 1}:")
    time_steps = 0

    while not done:
        env.render()
        action = np.argmax(q_table[state, :])
        state, reward, done, _, info = env.step(action)
        state = int(state)  # Updated line
        time_steps += 1

    env.render()
    print(f"Number of steps: {time_steps}\n")
