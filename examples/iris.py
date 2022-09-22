import gym
from urdfenvs import iris
import numpy as np


def main():
    env = gym.make("iris-rotor-vel-v0", dt=0.01, render=True)
    n_steps = 1000
    # set initial state [x, y, z, qx, qy, qz, qw]
    pos0 = np.array([-2.0, 0.0, 0.2, 0., 0., 0., 1.])
    action = np.array([2.0, 2.0, 2.0, 2.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    for i in range(n_steps):
        if i < 400:
            action = np.ones(4) * 835
        elif i < 500:
            action = np.ones(4) * 835 + np.array([0, 1, 0, 1]) * 2
        elif i < 600:
            action = np.ones(4) * 835 + np.array([0, 1, 0, 1]) * -2
        elif i < 700:
            action = np.ones(4) * 835 + np.array([0, 1, 1, 0]) * 2
        elif i < 800:
            action = np.ones(4) * 835 + np.array([0, 1, 1, 0]) * -2
        ob, _, _, _ = env.step(action)
        

if __name__ == "__main__":
    main()
