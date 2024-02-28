import numpy as np
import pytest

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from urdfenvs.scene_examples.obstacles import sphereObst1, urdfObst1, dynamicSphereObst3
from urdfenvs.robots.generic_urdf import GenericUrdfReacher

def test_static_obstacle():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = UrdfEnv(
        dt=0.01, robots=robots, render=False
    )
    # add sensor
    env.reset()
    env.add_obstacle(sphereObst1)
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    env.set_spaces()
    action = np.random.random(env.n())
    ob, *_ = env.step(action)
    ob = ob['robot_0']['ObstacleSensor']
    assert "obstacle_0" in ob
    assert isinstance(ob["obstacle_0"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacle_0"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacle_0"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacle_0"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacle_0"]["pose"]["position"],
        sphereObst1.position(t=env.t()),
        decimal=2,
    )
    env.close()


def test_dynamicObstacle():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = UrdfEnv(
        dt=0.01, robots=robots, render=False
    )

    # add sensor
    env.reset()
    env.add_obstacle(dynamicSphereObst3)
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    env.set_spaces()
    action = np.random.random(env.n())
    ob, *_ = env.step(action)
    ob = ob['robot_0']['ObstacleSensor']
    assert "obstacle_0" in ob
    assert isinstance(ob["obstacle_0"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacle_0"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacle_0"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacle_0"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacle_0"]["pose"]["position"],
        dynamicSphereObst3.position(t=env.t()),
        decimal=2,
    )
    env.close()


def test_shape_observation_space():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = UrdfEnv(
        dt=0.01, robots=robots, render=False
    )
    # add obstacle and sensor
    env.reset()
    env.add_obstacle(dynamicSphereObst3)
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    env.set_spaces()
    action = np.random.random(env.n())
    ob, *_ = env.step(action)
    ob = ob['robot_0']['ObstacleSensor']
    assert ob["obstacle_0"]["pose"]["position"].shape == (3, )
    assert ob["obstacle_0"]["pose"]["orientation"].shape == (4, )
    assert ob["obstacle_0"]["twist"]["linear"].shape == (3, )
    assert ob["obstacle_0"]["twist"]["angular"].shape == (3, )
    env.close()


@pytest.mark.skip(
    reason="Fails due to different position in pybullet and obstacle from motion planning scene"
)
def test_urdfObstacle(env):
    # add sensor
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    # change order
    env.add_obstacle(urdfObst1)
    action = np.random.random(env.n())
    ob, *_ = env.step(action)
    assert "obstacleSensor" in ob['robot_0']
    assert "2" in ob['robot_0']["obstacleSensor"]
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["position"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob['robot_0']["obstacleSensor"]["2"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob['robot_0']["obstacleSensor"]["pose"]["position"],
        dynamicSphereObst3.position(t=env.t()),
        decimal=2,
    )
    env.close()
