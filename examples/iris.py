import gym
from urdfenvs import iris
import numpy as np


def main():
    env = gym.make("iris-rotor-v0", dt=0.01, render=True)
    n_steps = 1000
    # set initial state [x, y, z, qx, qy, qz, qw]
    pos0 = np.array([-2.0, 0.0, 0.2, 0., 0., 0., 1.])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    for i in range(n_steps):
        if i < 500:
            action = np.array([1, 1, -1, -1]) * 835
        elif i < 550:
            action = np.array([1, 1, -1, -1]) * 835 + np.array([0, 1, 0, -1]) * 3
        elif i < 600:
            action = np.array([1, 1, -1, -1]) * 835 + np.array([0, 1, 0, -1]) * -3
        else:
            action = np.array([1, 1, -1, -1]) * 840
        ob, _, _, _ = env.step(action)
        

if __name__ == "__main__":
    main()