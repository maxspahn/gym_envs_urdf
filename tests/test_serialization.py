import gymnasium as gym
import numpy as np
import pytest
from urdfenvs.sensors.lidar import Lidar

from urdfenvs.scene_examples.obstacles import sphereObst1, dynamicSphereObst3
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def test_serialization():
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
    sensor = Lidar(4)
    env.add_sensor(sensor, [0])
    env.set_spaces()
    action = np.random.random(env.n())
    for i in range(10):
        ob, *_ = env.step(action)
    q_test = ob['robot_0']['joint_state']['position']
    lidar_test = ob['robot_0']['LidarSensor'][5]
 

    env.dump("temp_serialize.pkl")
    env.close()
    env_loaded = UrdfEnv.load("temp_serialize.pkl")
    env_loaded.reset()
    for _ in range(10):
        ob, *_ = env_loaded.step(action)
    q_loaded = ob['robot_0']["joint_state"]["position"]
    lidar_loaded = ob['robot_0']['LidarSensor'][5]

    assert q_test[0] == pytest.approx(q_loaded[0])
    assert lidar_test == pytest.approx(lidar_loaded)
    env_loaded.close()
    

