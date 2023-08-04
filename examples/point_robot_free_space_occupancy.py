import gymnasium as gym
import pprint
import numpy as np
from urdfenvs.scene_examples.obstacles import (
    sphereObst2,
    dynamicSphereObst1,
    cylinder_obstacle,
    wall_obstacles,
)

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.free_space_occupancy import FreeSpaceOccupancySensor
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_point_robot_with_freespacedecomp(
    n_steps=10000, render=False, obstacles=True, goal=True
):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=render)
    if obstacles:
        env.add_obstacle(sphereObst2)
        env.add_obstacle(cylinder_obstacle)
    val = 40
    free_space_decomp  = FreeSpaceOccupancySensor(
        'mobile_joint_theta',
        plotting_interval=1000,
        plotting_interval_fsd=100,
        max_radius=10,
        number_constraints=10,
        limits =  np.array([[-5, 5], [-5, 5], [0, 50/val]]),
        resolution = np.array([val + 1, val + 1, 5], dtype=int),
        interval=100,
    )
    env.add_sensor(free_space_decomp, robot_ids=[0])
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
        #print(ob['robot_0']['FreeSpaceDecompSensor'])
        # Access the lidar observation
        #_ = ob["robot_0"]["LidarSensor"]

        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_freespacedecomp(render=True)
