import numpy as np
from urdfenvs.sensors.lidar import Lidar

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.scene_examples.obstacles import sphereObst1, dynamicSphereObst3
from urdfenvs.robots.generic_urdf import GenericUrdfReacher


def test_full_sensor():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = UrdfEnv(
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
    ob, *_ = env.step(action)
    lidar_sensor_ob = ob['robot_0']['LidarSensor']
    assert isinstance(lidar_sensor_ob, np.ndarray)
    env.close()

