import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.scene_examples.goal import dynamicGoal
from urdfenvs.scene_examples.obstacles import dynamicSphereObst2
from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_panda(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="panda_with_gripper.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots,
        render=render,
        observation_checking=False,
    )
    env.add_goal(dynamicGoal)
    env.add_obstacle(dynamicSphereObst2)
    env.set_spaces()
    action = np.ones(env.n()) * 0.1
    ob = env.reset()
    print(f"Initial observation : {ob}")
    print("Starting episode")
    history = []
    for i in range(n_steps):
        if (int(i / 70)) % 2 == 0:
            action[7] = -0.02
            action[8] = -0.02
        else:
            action[7] = 0.02
            action[8] = 0.02
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_panda(render=True)
