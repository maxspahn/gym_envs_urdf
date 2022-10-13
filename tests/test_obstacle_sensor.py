import gym
import numpy as np
import pytest
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.lidar import Lidar

from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from examples.scene_objects.obstacles import sphereObst1, urdfObst1, dynamicSphereObst3
from examples.scene_objects.goal import goal1
from urdfenvs.robots.generic_urdf import GenericUrdfReacher

def test_full_sensor():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )
    # add sensor
    env.reset()
    env.add_obstacle(sphereObst1)
    sensor = FullSensor(goal_mask=['position'], obstacle_mask=['position', 'radius'])
    env.add_sensor(sensor, [0])
    sensor = Lidar(4)
    env.add_sensor(sensor, [0])
    env.add_goal(goal1)
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    ob = ob['robot_0']
    assert "obstacles" in ob
    assert "goals" in ob
    assert "rays" in ob
    assert isinstance(ob["obstacles"], list)
    assert isinstance(ob["obstacles"][0], list)
    assert isinstance(ob["obstacles"][0][0], np.ndarray)
    assert isinstance(ob["goals"], list)
    assert isinstance(ob["goals"][0][0], np.ndarray)
    assert isinstance(ob["rays"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacles"][0][0],
        sphereObst1.position(t=env.t()),
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        ob["goals"][0][0],
        goal1.position(t=env.t()),
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        ob["obstacles"][0][1],
        sphereObst1.radius(),
        decimal=2,
    )
    env.close()

def test_staticObstacle():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )
    # add sensor
    env.reset()
    env.add_obstacle(sphereObst1)
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    ob = ob['robot_0']
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
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )

    # add sensor
    env.reset()
    env.add_obstacle(dynamicSphereObst3)
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    ob = ob['robot_0']
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
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )
    # add obstacle and sensor
    env.reset()
    env.add_obstacle(dynamicSphereObst3)
    sensor = ObstacleSensor()
    env.add_sensor(sensor, [0])
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    ob = ob['robot_0']
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
    ob, _, _, _ = env.step(action)
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
