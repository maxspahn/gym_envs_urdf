import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_dual_arm(n_steps=5000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="dualArm.urdf", mode="vel"),
    ]
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_dual_arm(render=True)
