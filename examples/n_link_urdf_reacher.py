import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine
import numpy as np


def run_n_link_reacher(n_steps=1000, render=False, goal=True, obstacles=True):
    physics_engine = BulletPhysicsEngine(render)
    robots = [
        GenericUrdfReacher(physics_engine, urdf="nlink_3.urdf", mode="acc"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots,
    )
    n = env.n()
    action = np.ones(n) * 0.1
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_n_link_reacher(render=True)
