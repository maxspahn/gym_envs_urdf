import gym
from urdfenvs.robots.boxer import BoxerRobot
from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine
import numpy as np


def run_boxer(n_steps=1000, render=False, goal=True, obstacles=True):
    physics_engine = BulletPhysicsEngine(render)
    robots = [
        BoxerRobot(physics_engine, mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        physics_engine=physics_engine,
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.6, 0.8])
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
    run_boxer(render=True)
