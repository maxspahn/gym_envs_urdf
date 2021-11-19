import gym
import tiagoReacher
import time
import numpy as np


def main():
    env = gym.make('tiago-reacher-vel-v0', dt=0.01, render=True)
    defaultAction = np.random.rand(env.n()) * 0.0
    defaultAction[1] = 1.0
    defaultAction[0] = 0.5
    defaultAction[8] = 0.0
    n_episodes = 1
    n_steps = 10000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset()
        print("Starting episode")
        for i in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            print(ob)
            cumReward += reward


if __name__ == '__main__':
    main()
