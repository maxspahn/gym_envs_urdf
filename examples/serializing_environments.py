import gymnasium as gym
import numpy as np
from urdfenvs.sensors.lidar import Lidar

from urdfenvs.scene_examples.obstacles import sphereObst1, dynamicSphereObst3
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

#TODO: In loaded environments, sensors are not plotting debug shapes.


def serialize(file_name: str):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )
    # add sensor
    env.reset()
    env.add_obstacle(sphereObst1)
    env.add_obstacle(dynamicSphereObst3)
    env.add_goal(goal1)
    sensor = Lidar(4)
    env.add_sensor(sensor, [0])
    env.set_spaces()
    action = np.random.random(env.n())
    for i in range(10):
        ob, *_ = env.step(action)

    env.dump(file_name)
    env.close()

def load(file_name: str):
    env_loaded = UrdfEnv.load(file_name, render=True)
    return env_loaded

def run_serializing_example():
    serialize("point_robot_env.pkl")
    env = load("point_robot_env.pkl")
    for _ in range(1000000):
        env.step(np.zeros(env.n()))

if __name__ == "__main__":
    run_serializing_example()


    

