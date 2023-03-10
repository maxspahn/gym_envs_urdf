import gym
import numpy as np
import pytest
from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.scene_examples.obstacles import sphereObst1, dynamicSphereObst3
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def test_full_sensor():
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
    sensor = FullSensor(
        goal_mask=['position', 'is_primary_goal'],
        obstacle_mask=['velocity', 'position', 'size'],
        variance=0.0,
    )
    env.add_sensor(sensor, [0])
    env.add_goal(goal1)
    env.set_spaces()
    action = np.random.random(env.n())
    for _ in range(10):
        ob, _, _, _ = env.step(action)
    full_sensor_ob = ob['robot_0']['FullSensor']
    assert "obstacles" in full_sensor_ob.keys()
    assert "goals" in full_sensor_ob
    assert isinstance(full_sensor_ob["obstacles"], dict)
    assert isinstance(full_sensor_ob["obstacles"][2], dict)
    assert isinstance(full_sensor_ob["obstacles"][2]['position'], np.ndarray)
    assert isinstance(full_sensor_ob["goals"], dict)
    assert isinstance(full_sensor_ob["goals"][4]['position'], np.ndarray)
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]['velocity'],
        sphereObst1.velocity(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]['position'],
        sphereObst1.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][4]['position'],
        goal1.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][4]['is_primary_goal'],
        goal1.is_primary_goal(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]['size'],
        sphereObst1.radius(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][3]['velocity'],
        dynamicSphereObst3.velocity(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][3]['position'],
        dynamicSphereObst3.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][3]['size'],
        dynamicSphereObst3.radius(),
        decimal=2,
    )
    env.close()

