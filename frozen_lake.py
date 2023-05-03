import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def state_to_index(state):
    return int(state[0] * env.env.nrow + state[1])

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
training_episodes = 20000
testing_episodes = 300

# Creating the Frozen Lake environment
desc = generate_random_map(size=4)
env = gym.make("FrozenLake-v1", desc=desc, is_slippery=True)

# Q-table initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Training the agent
for episode in range(training_episodes):
    state = state_to_index(env.reset()[:2])
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        next_state, reward, done, _, _ = env.step(action)
        next_state = state_to_index(next_state[:2])
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        state = next_state

# Testing the trained agent
wins = 0
iterationInfo = []
cantIterations = 0

for episode in range(testing_episodes):
    print(f"Iteration no. {episode + 1}")
    cantIterations += 1

    state = state_to_index(env.reset()[:2])
    env.render()
    done = False

    while not done:
        action = np.argmax(q_table[state, :])
        state, reward, done, _, _ = env.step(action)
        state = state_to_index(state[:2])
        env.render()

        if reward == 1:
            print("Win\n")
            wins += 1
            iterationInfo.append(cantIterations)
            cantIterations = 0
            desc = generate_random_map(size=4)
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=True)
            break

        if done:
            print("Game Over\n")
            break

print(f"Number of wins: {wins}")
for x in range(len(iterationInfo)):
    print(f"{x + 1}: {iterationInfo[x]}")

env.close()
