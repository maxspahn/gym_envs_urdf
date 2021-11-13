import gym
from gym import envs
import pointRobotUrdf
import time
import numpy as np


def main():
    env = gym.make('pointRobotUrdf-vel-v0', dt=0.01, render=True)
    defaultAction = np.array([0.1, 0.3, 1.0])
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset()
        print("Starting episode")
        for i in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    main()
