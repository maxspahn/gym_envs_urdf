import gym
import numpy as np
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine
import warnings


def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    physics_engine = BulletPhysicsEngine(render)
    robots = [
        AlbertRobot(physics_engine, mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        physics_engine=physics_engine,
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(9)
    action[0] = 0.2
    action[1] = 0.0
    action[5] = -0.1
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
