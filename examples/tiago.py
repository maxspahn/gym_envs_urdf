import gym
import urdfenvs.tiagoReacher
import time
import numpy as np


def main():
    env = gym.make('tiago-reacher-vel-v0', dt=0.01, render=False)
    defaultAction = np.zeros(env.n())
    defaultAction[0:2] = np.array([0.0, 0.00])
    defaultAction[10] = 0.0
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    pos0 = np.zeros(20)
    pos0[0] = -1.7597e-1
    # base
    # pos0[0:3] = np.array([0.0, 1.0, -1.0])
    # torso
    pos0[3] = 0.1
    # head
    # pos0[4:6] = np.array([1.0, 0.0])
    # left arm
    # pos0[6:13] = np.array([0.0, 0.7, 0.2, 0.0, 0.0, 0.1, -0.1])
    # right arm
    # pos0[13:20] = np.array([-0.5, 0.2, 0.2, 0.0, 0.0, 0.1, -0.1])
    vel0 = np.zeros(19)
    # vel0[0:2] = np.array([0.0, 0.0])
    # vel0[5:12] = np.array([0.1, 0.1, 0.2, -0.1, 0.1, 0.2, 0.0])
    for e in range(n_episodes):
        ob = env.reset(pos=pos0, vel=vel0)
        print("base: ", ob['x'][0:3])
        print("torso: ", ob['x'][3])
        print("head: ", ob['x'][4:6])
        print("left arm: ", ob['x'][6:13])
        print("right arm: ", ob['x'][13:20])
        print("Starting episode")
        for i in range(n_steps):
            ob, reward, done, info = env.step(defaultAction)
            cumReward += reward


if __name__ == '__main__':
    main()
