import gym
import numpy as np
import pytest

from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from examples.scene_objects.obstacles import sphereObst1, urdfObst1, dynamicSphereObst3
from urdfenvs.robots.generic_urdf import GenericUrdfReacher


@pytest.fixture
def pointRobotEnv():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )
    env.reset()
    env.add_obstacle(sphereObst1)
    env.add_obstacle(dynamicSphereObst3)
    return env


def test_staticObstacle(pointRobotEnv):

    # add sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor, [0])
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)
    assert "obstacleSensor" in ob['robot_0']
    assert "2" in ob['robot_0']["obstacleSensor"]
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob['robot_0']["obstacleSensor"]["2"]["pose"]["position"],
        sphereObst1.position(t=pointRobotEnv.t()),
        decimal=2,
    )


def test_dynamicObstacle(pointRobotEnv):

    # add sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor, [0])
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)
    ob, _, _, _ = pointRobotEnv.step(action)
    assert "obstacleSensor" in ob['robot_0']
    assert "3" in ob['robot_0']["obstacleSensor"]
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob['robot_0']["obstacleSensor"]["3"]["pose"]["position"],
        dynamicSphereObst3.position(t=pointRobotEnv.t()),
        decimal=2,
    )


def test_shape_observation_space(pointRobotEnv):
    # add obstacle and sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor, [0])
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)

    assert ob['robot_0']["obstacleSensor"]["2"]["pose"]["position"].shape == (3, )
    assert ob['robot_0']["obstacleSensor"]["2"]["pose"]["orientation"].shape == (4, )
    assert ob['robot_0']["obstacleSensor"]["2"]["twist"]["linear"].shape == (3, )
    assert ob['robot_0']["obstacleSensor"]["2"]["twist"]["angular"].shape == (3, )


@pytest.mark.skip(
    reason="Fails due to different position in pybullet and obstacle from motion planning scene"
)
def test_urdfObstacle(pointRobotEnv):
    # add sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor, [0])
    # change order
    pointRobotEnv.add_obstacle(urdfObst1)
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)
    assert "obstacleSensor" in ob['robot_0']
    assert "2" in ob['robot_0']["obstacleSensor"]
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob['robot_0']["obstacleSensor"]["pose"]["position"],
        dynamicSphereObst3.position(t=pointRobotEnv.t()),
        decimal=2,
    )
