import gym
import urdfenvs.n_link_urdf_reacher
import numpy as np


def run_n_link_reacher(n_steps=1000, render=False, goal=True, obstacles=True):
    n = 3
    env = gym.make("nLink-urdf-reacher-acc-v0", n=n, dt=0.01, render=render)
    action = np.ones(n) * 0.1
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    return history


if __name__ == "__main__":
    main()
    run_n_link_reacher(render=True)
