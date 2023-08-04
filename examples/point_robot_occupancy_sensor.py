import gymnasium as gym
import numpy as np

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.occupancy_sensor import OccupancySensor
from urdfenvs.scene_examples.obstacles import (
    cylinder_obstacle,
    sphereObst2,
    sphereObst1,
    dynamicSphereObst1,
)


def get_index_from_coordinates(point, mesh) -> tuple:
    distances = np.linalg.norm(mesh - point, axis=3)
    return np.unravel_index(np.argmin(distances), mesh.shape[:-1])

def evaluate_occupancy(point, mesh, occupancy, resolution) -> int:
    index = list(get_index_from_coordinates(point, mesh))
    return occupancy[tuple(index)]




def run_point_robot_with_occupancy_sensor(n_steps=10, render=False, obstacles=True, goal=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    env.add_obstacle(sphereObst2)
    env.add_obstacle(cylinder_obstacle)
    env.add_obstacle(sphereObst1)
    env.add_obstacle(dynamicSphereObst1)

    # add sensor
    val = 40
    sensor = OccupancySensor(
        limits =  np.array([[-5, 5], [-5, 5], [0, 50/val]]),
        resolution = np.array([val + 1, val + 1, 5], dtype=int),
        interval=100,
        plotting_interval=100,
    )

    env.add_sensor(sensor, [0])
    # Set spaces AFTER all components have been added.
    env.set_spaces()
    defaultAction = np.array([0.5, -0.2, 0.0])
    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    initial_observations = []
    ob, _ = env.reset(pos=pos0, vel=vel0)
    initial_observations.append(ob)
    env.add_debug_shape(
        (0.2, -0.3, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        size=[0.3],
        rgba_color=[0.0, 0.0, 0.0, 0.3],
    )

    history = []
    for _ in range(n_steps):
        action = defaultAction
        ob, *_ = env.step(action)
        point = np.append(ob['robot_0']['joint_state']['position'][0:2], 0.0)
        occupancy = ob['robot_0']['Occupancy']
        occupancy_eval = evaluate_occupancy(point, sensor.mesh(), occupancy, [0.2, 0.2, 1])
        print(occupancy_eval)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_occupancy_sensor(render=True, n_steps=300)
