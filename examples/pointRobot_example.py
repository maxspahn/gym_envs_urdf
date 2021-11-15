import gym
from gym import envs
import pointRobotUrdf
import time
import numpy as np


def main():
    env = gym.make('pointRobotUrdf-acc-v0', dt=0.05, render=True)
    defaultAction = np.array([0.0, 0.0])
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    pos0 = np.array([1.0, 0.1])
    vel0 = np.array([1.0, 0.0])
    for e in range(n_episodes):
        ob = env.reset(pos=pos0, vel=vel0)
        env.setWalls(limits=[[-3, -2], [3, 2]])
        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t += env.dt()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    main()
