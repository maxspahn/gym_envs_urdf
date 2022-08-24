import gym
import numpy as np
import pytest

from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from examples.scene_objects.obstacles import sphereObst1, urdfObst1, dynamicSphereObst3
import urdfenvs.point_robot_urdf


@pytest.fixture
def point_robot_env():
    import urdfenvs.point_robot_urdf

    env = gym.make("pointRobotUrdf-vel-v0", render=False, dt=0.01)
    _ = env.reset()
    return env


def test_static_obstacle(point_robot_env):
    point_robot_env.add_obstacle(sphereObst1)

    # add sensor
    sensor = ObstacleSensor()
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)
    assert "obstacleSensor" in ob
    assert "2" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["2"]["pose"]["position"],
        sphereObst1.position(),
        decimal=2,
    )


def test_dynamic_obstacle(point_robot_env):
    point_robot_env.add_obstacle(dynamicSphereObst3)

    # add sensor
    sensor = ObstacleSensor()
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)
    assert "obstacleSensor" in ob
    assert "2" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["2"]["pose"]["position"],
        dynamicSphereObst3.position(t=point_robot_env.t()),
        decimal=2,
    )


def test_shape_observation_space(point_robot_env):
    # add obstacle and sensor
    point_robot_env.add_obstacle(sphereObst1)
    sensor = ObstacleSensor()
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)

    assert ob["obstacleSensor"]["2"]["pose"]["position"].shape == (3, )
    assert ob["obstacleSensor"]["2"]["pose"]["orientation"].shape == (4, )
    assert ob["obstacleSensor"]["2"]["twist"]["linear"].shape == (3, )
    assert ob["obstacleSensor"]["2"]["twist"]["angular"].shape == (3, )


@pytest.mark.skip(
    reason="Fails due to different position in pybullet and "\
            "obstacle from motion planning scene"
)

def test_urdf_obstacle(point_robot_env):
    # add sensor
    sensor = ObstacleSensor()
    point_robot_env.add_sensor(sensor)
    # change order
    point_robot_env.add_obstacle(urdfObst1)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)
    assert "obstacleSensor" in ob
    assert "2" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["pose"]["position"],
        dynamicSphereObst3.position(t=point_robot_env.t()),
        decimal=2,
    )
