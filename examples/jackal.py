import gym
from urdfenvs.robots.jackal import JackalRobot
import numpy as np


def run_jackal(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        JackalRobot(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([1.0, 0.50])
    pos0 = np.array([1.0, 0.2, -1.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    env.add_walls()
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_jackal(render=True, n_steps=1000)
