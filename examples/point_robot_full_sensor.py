import gym
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.scene_examples.obstacles import (
    sphereObst1,
    movable_obstacle,
    dynamicSphereObst3,
)
import numpy as np

from gym.wrappers.flatten_observation import FlattenObservation



def run_point_robot_with_obstacle_sensor(n_steps=10, render=False, obstacles=True, goal=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    #env.add_obstacle(sphereObst1)
    env.add_obstacle(dynamicSphereObst3)
    env.add_obstacle(movable_obstacle)
    env.add_goal(goal1)

    # add sensor
    sensor = FullSensor(['position'], ['position', 'size'], variance=0.0)
    env.add_sensor(sensor, [0])
    # Set spaces AFTER all components have been added.
    defaultAction = np.array([0.5, -0.2, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    initial_observations = []
    for e in range(2):
        env.from_file(f"env_{e}.yaml")
        env.set_spaces()
        env = FlattenObservation(env)
        ob = env.reset(pos=pos0, vel=vel0)
        initial_observations.append(ob)
        print(f"Initial observation : {ob}")
        #assert np.array_equal(initial_observations[0], ob)

        history = []
        for _ in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            # In observations, information about obstacles is stored in ob['obstacleSensor']
            history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_obstacle_sensor(render=True, n_steps=3)
