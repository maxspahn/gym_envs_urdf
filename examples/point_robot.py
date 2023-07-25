import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import numpy as np

from urdfenvs.scene_examples.obstacles import *
from urdfenvs.scene_examples.goal import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_point_robot(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.1, 0.0, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    if obstacles:
        env.add_obstacle(sphereObst1)
        env.add_obstacle(movable_obstacle)
        env.add_obstacle(urdfObst1)
        env.add_obstacle(dynamicSphereObst3)
        env.add_obstacle(cylinder_obstacle)
    if goal:
        env.add_goal(splineGoal)
    history = []
    env.reconfigure_camera(2.0, 0.0, -90.01, (0, 0, 0))
    for _ in range(n_steps):
        ob, _, terminated, _, info  = env.step(action)
        if terminated:
            print(info)
            break
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot(render=True)
