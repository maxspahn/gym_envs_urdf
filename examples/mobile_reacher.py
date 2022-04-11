import gym
import urdfenvs.mobile_reacher
import numpy as np


def main():
    env = gym.make("mobile-reacher-acc-v0", dt=0.01, render=True, gripper=False)

    action = np.zeros(10)
    action[0] = 0.1
    action[5] = -0.0
    action[-1] = 3.5
    n_steps = 100000
    ob = env.reset()
    print(f"Initial observation : {ob}")
    for i in range(n_steps):
        if (int(i / 100)) % 2 == 0:
            action[-1] = -1.0
        else:
            action[-1] = 1.0
        ob, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
