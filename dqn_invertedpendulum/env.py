import mujoco
import gym
import numpy as np
from test1 import dqnagent as agent
import random

env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]

action_dim=2

dqn_agent = agent(statd=state_dim, actd=action_dim)

num_episodes = 1000
maxre=0
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = dqn_agent.selection(state)

        next_state, reward, done,_, _= env.step(action)

        dqn_agent.store(state, action, next_state, reward, done)

        dqn_agent.play()

        state = next_state
        total_reward += reward
        if total_reward>maxre:
            maxre=total_reward

        if done:
            env.reset()
            break

    dqn_agent.epsilon_decay()

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

print(maxre)
