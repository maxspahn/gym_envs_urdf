import gym
import nLinkUrdfReacher
import time
import numpy as np
import matplotlib.pyplot as plt

from nLinkUrdfReacher.resources.accController import NLinkUrdfAccController

def main():
    n = 4
    k = np.ones(n) * 0.0
    accController = NLinkUrdfAccController(n, k)
    env = gym.make('nLink-urdf-reacher-tor-v0', n=n, dt=0.01, render=True)
    defaultAction = np.ones(n) * 0.00
    n_episodes = 1
    n_steps = 2000
    cumReward = 0.0
    qs = np.zeros((n_steps, n))
    qdots = np.zeros((n_steps, n))
    for e in range(n_episodes):
        ob = env.reset()
        for i in range(5):
            action = np.ones(n) * 0.2
            #ob, reward, done, info = env.step(1.0 * env.action_space.sample())
            ob, reward, done, info = env.step(action)
        print("Starting episode")
        for i in range(n_steps):
            time.sleep(env._dt)
            action = env.action_space.sample()
            q = ob[0:n]
            qs[i, :] = q
            qdot = ob[n:2*n]
            qdots[i, :] = qdot
            qddot = defaultAction
            action = -1.0 * np.array(accController.control(q, qdot, qddot))
            print("tau casadi : ", action)
            ob, reward, done, info = env.step(action)
            cumReward += reward

    #plt.plot(qs)
    plt.plot(qdots)
    plt.legend(["q0", "q1", "qdot0", "qdot1"])
    plt.show()


if __name__ == '__main__':
    main()
