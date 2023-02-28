import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from urdfenvs.scene_examples.obstacles import (
    sphereObst1,
    urdfObst1,
    dynamicSphereObst3,
)
import numpy as np


def run_point_robot_with_obstacle_sensor(n_steps=10, render=False, obstacles=True, goal=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    defaultAction = np.array([0.1, 0.0, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    for _ in range(2):
        ob = env.reset(pos=pos0, vel=vel0)
        print(f"Initial observation : {ob}")

        # add obstacles
        env.add_obstacle(sphereObst1)
        env.add_obstacle(dynamicSphereObst3)

        # add sensor
        sensor = FullSensor(['position'], ['position', 'size'])
        env.add_sensor(sensor, [0])

        history = []
        for _ in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            print(ob)
            # In observations, information about obstacles is stored in ob['obstacleSensor']
            history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_obstacle_sensor(render=True)