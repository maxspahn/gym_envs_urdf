import gymnasium as gym
import numpy as np
from urdfenvs.scene_examples.obstacles import (
    sphereObst1,
    sphereObst2,
    urdfObst1,
    dynamicSphereObst3,
    wall_obstacles,
)

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.lidar import Lidar
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_point_robot_with_lidar(
    n_steps=1000, render=False, obstacles=True, goal=True
):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=render)
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    if obstacles:
        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(urdfObst1)
        env.add_obstacle(dynamicSphereObst3)
    number_lidar_rays = 64
    lidar = Lidar("lidar_sensor_joint", nb_rays=number_lidar_rays, raw_data=False)
    env.add_sensor(lidar, robot_ids=[0])
    env.set_spaces()

    action = np.array([0.1, -0.2, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    # Setup for showing LiDAR detections
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        # Access the lidar observation
        #_ = ob["robot_0"]["LidarSensor"]

        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_lidar(render=True)
