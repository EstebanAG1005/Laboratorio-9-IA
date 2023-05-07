import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time


def reset_environment_and_update_wins(env, reward, wins, iterationInfo, cantIterations):
    if reward == 1:
        print("Win\n")
        wins += 1
        iterationInfo.append(cantIterations)
        cantIterations = 0
    else:
        print("Game Over\n")

    return env, wins, iterationInfo, cantIterations


# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999
training_episodes = 20000
testing_episodes = 300

# Creating the Frozen Lake environment
desc = generate_random_map(size=4)
env = gym.make("FrozenLake-v1", render_mode="human", desc=desc, is_slippery=True)
env.metadata["render_fps"] = 30

# Q-table initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Training the agent
for episode in range(training_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        next_state, reward, done, _, _ = env.step(action)

        # Update the modified_reward calculation
        if done and reward == 0:
            modified_reward = -0.5
        else:
            modified_reward = reward - 0.01

        q_table[state, action] += alpha * (
            modified_reward
            + gamma * np.max(q_table[next_state, :])
            - q_table[state, action]
        )
        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon

    print(f"Training progress: Episode {episode + 1} of {training_episodes}")

# Testing the trained agent
wins = 0
iterationInfo = []
cantIterations = 0

for episode in range(testing_episodes):
    print(f"Iteration no. {episode + 1}")
    cantIterations += 1

    state = env.reset()[0]
    env.render()
    done = False

    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        env.render()
        time.sleep(0.5)

        if done:
            (
                env,
                wins,
                iterationInfo,
                cantIterations,
            ) = reset_environment_and_update_wins(
                env, reward, wins, iterationInfo, cantIterations
            )
            break

print(f"Number of wins: {wins}")
for x in range(len(iterationInfo)):
    print(f"{x + 1}: {iterationInfo[x]}")

env.close()
