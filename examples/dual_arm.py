import gym
import urdfenvs.dual_arm
import numpy as np


def run_dual_arm(n_steps=5000, render=False, goal=True, obstacles=True):
    env = gym.make("dual-arm-vel-v0", dt=0.01, render=render)
    action = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    return history


if __name__ == "__main__":
    run_dual_arm(render=True)
