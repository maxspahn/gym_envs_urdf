import gym
import urdfenvs.albertReacher
import numpy as np
import warnings


def main():
    env = gym.make('albert-reacher-vel-v0', dt=0.01, render=True)
    #env = gym.make('albert-v0', dt=0.01, render=True)
    defaultAction = np.zeros(9)
    defaultAction[0] = 0.2
    defaultAction[1] = 0.0
    defaultAction[5] = -0.0
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset()
        print("Starting episode")
        for i in range(n_steps):
            action = env.action_space.sample()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    showWarnings = False
    warningFlag = 'default' if showWarnings else 'ignore'
    with warnings.catch_warnings():
        warnings.filterwarnings(warningFlag)
        main()
