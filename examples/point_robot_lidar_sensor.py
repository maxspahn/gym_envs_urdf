import gym
import urdfenvs.point_robot_urdf
from urdfenvs.sensors.lidar import Lidar
import numpy as np

obstacles = True


def main():
    env = gym.make("pointRobotUrdf-vel-v0", dt=0.05, render=True, flatten_observation=False)
    lidar = Lidar(4, nb_rays=4, raw_data=False)
    env.add_sensor(lidar)
    action = np.array([0.1, 0.0, 0.0])
    n_steps = 100000
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
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
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        print(ob['lidarSensor'])


if __name__ == "__main__":
    main()
