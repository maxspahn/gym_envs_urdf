import gym
import nLinkUrdfReacher
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import urdf2casadi.urdfparser as u2c

from nLinkUrdfReacher.resources.accController import NLinkUrdfAccController
import nLinkUrdfReacher


def main():
    n = 4
    nLinkRobot = u2c.URDFparser()
    urdf_file = os.path.dirname(nLinkUrdfReacher.__file__) + "/resources/nlink_" + str(n) + ".urdf"
    nLinkRobot.from_file(urdf_file)

    root = "panda_link0"
    tip = "panda_link" + str(n+1)

    gravity = [0, 0, -10.0]
    id_sym = nLinkRobot.get_inverse_dynamics_rnea(root, tip, gravity)
    k = np.ones(n) * 0.0
    accController = NLinkUrdfAccController(n, k)
    env = gym.make('nLink-urdf-reacher-tor-v0', n=n, dt=0.01, render=True)
    defaultAction = np.ones(n) * -0.02
    n_episodes = 2
    n_steps = 4000
    cumReward = 0.0
    qs = np.zeros((n_steps, n))
    qdots = np.zeros((n_steps, n))
    qs2 = np.zeros((n_steps, n))
    qdots2 = np.zeros((n_steps, n))
    for e in range(n_episodes):
        ob = env.reset()
        print("Starting episode")
        for i in range(5):
            action = np.ones(n) * 0.0
            #ob, reward, done, info = env.step(1.0 * env.action_space.sample())
            ob, reward, done, info = env.step(action)
        for i in range(n_steps):
            #time.sleep(env._dt)
            if e == 0:
                q = ob[0:n]
                qs[i, :] = q
                qdot = ob[n:2*n]
                qdots[i, :] = qdot
                qddot = defaultAction
                action = -1.0 * np.array(accController.control(q, qdot, qddot))
                ob, reward, done, info = env.step(action)
            if e == 1:
                print("Second")
                q = ob[0:n]
                qs2[i, :] = q
                qdot = ob[n:2*n]
                qdots2[i, :] = qdot
                qddot = defaultAction
                action = id_sym(q, qdot, qddot)
                ob, reward, done, info = env.step(action)
            cumReward += reward

    #plt.plot(qs)
    plt.plot(qdots, 'o')
    plt.plot(qdots2)
    leg1 = ["qdot" + str(i) + "_1" for i in range(n)]
    leg2 = ["qdot" + str(i) + "_2" for i in range(n)]
    plt.legend(leg1 + leg2)
    plt.show()


if __name__ == '__main__':
    main()
