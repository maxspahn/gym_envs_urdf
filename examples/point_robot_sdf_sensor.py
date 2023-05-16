import gym
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.sdf_sensor import SDFSensor
from urdfenvs.scene_examples.obstacles import (
    cylinder_obstacle,
    sphereObst1,
)

import numpy as np

def get_index_from_coordinates(point, mesh) -> tuple:
    distances = np.linalg.norm(mesh - point, axis=3)
    return np.unravel_index(np.argmin(distances), mesh.shape[:-1])

def evaluate_sdf(point, mesh, sdf, resolution) -> tuple:
    index = list(get_index_from_coordinates(point, mesh))
    value = sdf[tuple(index)]
    gradient = np.zeros(3)
    for dim in range(3):
        lower_index = tuple(index[:dim] + [index[dim] - 1] + index[dim+1:])
        upper_index = tuple(index[:dim] + [index[dim] + 1] + index[dim+1:])
        breakpoint()

        if lower_index[dim] < 0 or upper_index[dim] >= sdf.shape[dim]:
            gradient[dim] = 0.0
        else:
            gradient[dim] = (sdf[upper_index] - sdf[lower_index]) / resolution[dim]

    return value, gradient




def run_point_robot_with_sdf_sensor(n_steps=10, render=False, obstacles=True, goal=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    #env.add_obstacle(sphereObst1)
    env.add_obstacle(cylinder_obstacle)
    env.add_obstacle(sphereObst1)

    # add sensor
    sensor = SDFSensor(
                 limits =  np.array([[-5, 5], [-5, 5], [0, 0]]),
                 resolution = np.array([51, 51, 1], dtype=int),
    )

    env.add_sensor(sensor, [0])
    # Set spaces AFTER all components have been added.
    env.set_spaces()
    defaultAction = np.array([0.5, -0.2, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    initial_observations = []
    for _ in range(2):
        ob = env.reset(pos=pos0, vel=vel0)
        initial_observations.append(ob)
        sdf = ob['robot_0']['SDFSensor']
        indices = get_index_from_coordinates(np.array([2.0, -1.0, 0.0]), sensor.mesh())
        point = np.array([1.0, 0.1, 0.0])
        sdf_eval = evaluate_sdf(np.array([1.0, 0.1, 0.0]), sensor.mesh(), sdf, [0.2, 0.2, 1])
        print(sdf_eval)
        breakpoint()
        #print(f"Initial observation : {ob}")
        #assert np.array_equal(initial_observations[0], ob)

        history = []
        for _ in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_sdf_sensor(render=True, n_steps=300)
