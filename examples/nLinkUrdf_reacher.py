import gym
import urdfenvs.nLinkUrdfReacher
import numpy as np


def main():
    n = 4
    #env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    env = gym.make('nLink-urdf-reacher-vel-v0', n=n, dt=0.01, render=True)
    defaultAction = np.ones(n) * 0.1
    n_episodes = 1
    n_steps = 1
    cumReward = 0.0
    pos0 = np.array([1.0, 0.0, 0.0, 0.1])
    vel0 = np.array([0.0, 0.0, 0.0, 0.0])
    for e in range(n_episodes):
        ob = env.reset(pos=pos0, vel=vel0)
        print(ob['x'])
        print("Starting episode")
        for i in range(n_steps):
            action = env.action_space.sample()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            print(ob['x'])
            cumReward += reward


if __name__ == '__main__':
    main()
