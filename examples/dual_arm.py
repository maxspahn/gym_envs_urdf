import gym
import urdfenvs.dual_arm
import numpy as np


def main():
    env = gym.make("dual-arm-vel-v0", dt=0.01, render=True)
    action = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
    n_steps = 5000
    ob = env.reset()
    print(f"Initial observation : {ob}")
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
