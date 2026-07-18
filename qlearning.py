import numpy as np 
import gymnasium as gym
import matplotlib.pyplot as plt

# state, info = env.reset() #returns the initial state and info
# print(env.render())
# print(env.observation_space.n, env.action_space.n)   #the agent has 4 actions to choose from in each state
# print(state, info)


# next_state, reward, terminated, truncated, info = env.step(2) #returns the next state, reward, terminated, truncated, and info
# next_state, reward, terminated, truncated, info = env.step(1) 
# next_state, reward, terminated, truncated, info = env.step(1) 
# print(next_state, reward, terminated, truncated, info)

env = gym.make("FrozenLake-v1", is_slippery=False)  # flip to True once this works

# --- lives across the whole run ---
Q = np.zeros((env.observation_space.n, env.action_space.n))  # (16, 4)
alpha = 0.1        # learning rate: how much to trust each new sample
gamma = 0.99       # discount: how much future reward matters
epsilon = 0.9      # start fully random
eps_min = 0.01
eps_decay = 0.999
n_episodes = 2000

results = []       # 1 if episode reached goal, else 0

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # epsilon-greedy: explore with prob epsilon, else exploit
        if np.random.random() < epsilon:
            action = env.action_space.sample()          # explore
        else:
            action = np.argmax(Q[state])                # exploit: INDEX of best action

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # the Q-learning update: Q(s,a) += alpha * (TD target - old estimate)
        # np.max = VALUE of best next action (argmax would be the bug here)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state   # only AFTER the update — it needs both s and s'

    # once per episode, not per step
    epsilon = max(eps_min, epsilon * eps_decay)
    results.append(reward)   # final reward is 1 iff we reached the goal

    if (episode + 1) % 500 == 0:
        recent = np.mean(results[-100:])
        print(f"episode {episode+1:5d}  |  success rate (last 100): {recent:.2f}  |  epsilon: {epsilon:.3f}")

print("\nFinal Q-table:")
print(np.round(Q, 3))
print(np.shape(Q))

# policy as arrows on the 4x4 grid
arrows = np.array(list("←↓→↑"))[np.argmax(Q, axis=1)].reshape(4, 4)
print("\nLearned policy:")
print(arrows)