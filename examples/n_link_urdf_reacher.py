import gym
import urdfenvs.n_link_urdf_reacher
import numpy as np


def main():
    n = 3
    # env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    env = gym.make("nLink-urdf-reacher-acc-v0", n=n, dt=0.01, render=True)
    action = np.ones(n) * 0.1
    n_steps = 1000
    ob = env.reset()
    print(f"Initial observation : {ob}")
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
