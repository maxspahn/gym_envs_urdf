import gym
import numpy as np
import pytest
from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.scene_examples.obstacles import sphereObst1, dynamicSphereObst3
from urdfenvs.scene_examples.goal import goal1
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
    env.add_obstacle(dynamicSphereObst3)
    sensor = FullSensor(
        goal_mask=['position', 'is_primary_goal'],
        obstacle_mask=['velocity', 'position', 'radius']
    )
    env.add_sensor(sensor, [0])
    env.add_goal(goal1)
    action = np.random.random(env.n())
    for _ in range(10):
        ob, _, _, _ = env.step(action)
    full_sensor_ob = ob['robot_0']['FullSensor']
    assert "obstacles" in full_sensor_ob.keys()
    assert "goals" in full_sensor_ob
    assert isinstance(full_sensor_ob["obstacles"], list)
    assert isinstance(full_sensor_ob["obstacles"][0], list)
    assert isinstance(full_sensor_ob["obstacles"][0][0], np.ndarray)
    assert isinstance(full_sensor_ob["goals"], list)
    assert isinstance(full_sensor_ob["goals"][0][0], np.ndarray)
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][0][0],
        sphereObst1.velocity(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][0][1],
        sphereObst1.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][0][0],
        goal1.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][0][1],
        goal1.is_primary_goal(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][0][2],
        sphereObst1.radius(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][1][0],
        dynamicSphereObst3.velocity(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][1][1],
        dynamicSphereObst3.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][1][2],
        dynamicSphereObst3.radius(),
        decimal=2,
    )
    env.close()

