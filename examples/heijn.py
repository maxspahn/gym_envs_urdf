import gymnasium as gym
import numpy as np
from urdfenvs.scene_examples.obstacles import *
from urdfenvs.scene_examples.goal import *

from urdfenvs.robots.generic_urdf import GenericUrdfReacher

def run_point_robot(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="heijn_robot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.1, 0.0, 0.01])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    if obstacles:
        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst3)
    if goal:
        env.add_goal(splineGoal)
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot(render=True)
