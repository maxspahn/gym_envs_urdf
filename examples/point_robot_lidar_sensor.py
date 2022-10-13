import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.lidar import Lidar
import numpy as np


def run_point_robot_with_lidar(n_steps=1000, render=False, obstacles=True, goal=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.1, 0.0, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    lidar = Lidar(4, nb_rays=4, raw_data=False)
    ob = env.reset(pos=pos0, vel=vel0)
    env.add_sensor(lidar, robot_ids=[0])
    print(f"Initial observation : {ob}")
    env.add_walls()
    if obstacles:
        from examples.scene_objects.obstacles import (
            sphereObst1,
            sphereObst2,
            urdfObst1,
            dynamicSphereObst3,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(urdfObst1)
        env.add_obstacle(dynamicSphereObst3)
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_lidar(render=True)
