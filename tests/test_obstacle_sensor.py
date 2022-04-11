import gym
import numpy as np
import pytest

from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from examples.scene_objects.obstacles import sphereObst1, urdfObst1, dynamicSphereObst3
import urdfenvs.point_robot_urdf


@pytest.fixture
def pointRobotEnv():
    import urdfenvs.point_robot_urdf

    env = gym.make("pointRobotUrdf-vel-v0", render=False, dt=0.01)
    _ = env.reset()
    return env


def test_staticObstacle(pointRobotEnv):
    pointRobotEnv.add_obstacle(sphereObst1)

    # add sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor)
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)
    assert "obstacleSensor" in ob
    assert "2" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["2"]["pose"]["position"],
        sphereObst1.position(t=pointRobotEnv.t()),
        decimal=2,
    )


def test_dynamicObstacle(pointRobotEnv):
    pointRobotEnv.add_obstacle(dynamicSphereObst3)

    # add sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor)
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)
    assert "obstacleSensor" in ob
    assert "2" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["2"]["pose"]["position"],
        dynamicSphereObst3.position(t=pointRobotEnv.t()),
        decimal=2,
    )


def test_shape_observation_space(pointRobotEnv):
    # add obstacle and sensor
    pointRobotEnv.add_obstacle(sphereObst1)
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor)
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)

    assert ob["obstacleSensor"]["2"]["pose"]["position"].shape == (3, )
    assert ob["obstacleSensor"]["2"]["pose"]["orientation"].shape == (4, )
    assert ob["obstacleSensor"]["2"]["twist"]["linear"].shape == (3, )
    assert ob["obstacleSensor"]["2"]["twist"]["angular"].shape == (3, )


@pytest.mark.skip(
    reason="Fails due to different position in pybullet and obstacle from motion planning scene"
)
def test_urdfObstacle(pointRobotEnv):
    # add sensor
    sensor = ObstacleSensor()
    pointRobotEnv.add_sensor(sensor)
    # change order
    pointRobotEnv.add_obstacle(urdfObst1)
    action = np.random.random(pointRobotEnv.n())
    ob, _, _, _ = pointRobotEnv.step(action)
    assert "obstacleSensor" in ob
    assert "2" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["pose"]["position"],
        dynamicSphereObst3.position(t=pointRobotEnv.t()),
        decimal=2,
    )
