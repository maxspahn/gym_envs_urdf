import gym
from urdfenvs import iris
import numpy as np


def main():
    env = gym.make("iris-vel-v0", dt=0.01, render=True)
    action = np.array([1.1, 0.1, 1.1])
    n_steps = 1000
    pos0 = np.array([-1.0, 0.2, -0.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        

if __name__ == "__main__":
    main()