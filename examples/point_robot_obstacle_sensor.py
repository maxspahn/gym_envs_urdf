import gym
import urdfenvs.point_robot_urdf
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from examples.scene_objects.obstacles import (
    sphereObst1,
    urdfObst1,
    dynamicSphereObst3,
)
import numpy as np


def main():
    env = gym.make("pointRobotUrdf-vel-v0", dt=0.05, render=True)

    defaultAction = np.array([0.1, 0.0, 0.0])
    n_episodes = 1
    n_steps = 100000
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")

    # add obstacles
    env.add_obstacle(sphereObst1)
    env.add_obstacle(urdfObst1)
    env.add_obstacle(dynamicSphereObst3)

    # add sensor
    sensor = ObstacleSensor()
    env.add_sensor(sensor)

    for e in range(n_episodes):

        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t += env.dt()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            # In observations, information about obstacles is stored in ob['obstacleSensor']
            print(ob["obstacleSensor"])


if __name__ == "__main__":
    main()
