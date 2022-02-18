import gym
import urdfenvs.pointRobotUrdf
from urdfenvs.sensors.obstacleSensor import ObstacleSensor
from examples.sceneObjects.obstacles import sphereObst1, urdfObst1, dynamicSphereObst3
import numpy as np


def main():
    env = gym.make('pointRobotUrdf-vel-v0', dt=0.05, render=True)

    defaultAction = np.array([0.1, 0.0, 0.0])
    n_episodes = 1
    n_steps = 100000
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)

    # add obstacles
    env.addObstacle(sphereObst1)
    env.addObstacle(urdfObst1)
    env.addObstacle(dynamicSphereObst3)

    # add sensor
    sensor = ObstacleSensor()
    env.addSensor(sensor)

    for e in range(n_episodes):

        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t += env.dt()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            # In observations, information about obstacles is stored in ob['obstacleSensor']
            print(ob['obstacleSensor'])


if __name__ == '__main__':
    main()
