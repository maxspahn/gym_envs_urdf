import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_mobile_reacher(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel"),
    ]
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=render, num_sub_steps=200,
    )
    action = np.zeros(env.n())
    action[0] = 0.1
    action[5] = -0.0
    action[-1] = 3.5
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    for i in range(n_steps):
        if (int(i / 100)) % 2 == 0:
            action[-1] = -0.01
            action[-2] = -0.01
        else:
            action[-1] = 0.01
            action[-2] = 0.01
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_mobile_reacher(render=True)
