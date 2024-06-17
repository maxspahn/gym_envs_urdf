import os
import shutil

import numpy as np
import gymnasium as gym

from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.scene_examples.obstacles import sphereObst1, dynamicSphereObst3, movable_obstacle
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot


def test_full_sensor():
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=False)
    # add sensor
    env.reset()
    env.add_obstacle(sphereObst1)
    env.add_obstacle(dynamicSphereObst3)
    env.add_obstacle(movable_obstacle)
    sensor = FullSensor(
        goal_mask=["position", "is_primary_goal"],
        obstacle_mask=["velocity", "position", "size"],
        variance=0.0,
    )
    env.add_sensor(sensor, [0])
    env.add_goal(goal1)
    env.set_spaces()
    action = np.random.random(env.n())
    for _ in range(10):
        ob, *_ = env.step(action)
    full_sensor_ob = ob["robot_0"]["FullSensor"]
    assert "obstacles" in full_sensor_ob.keys()
    assert "goals" in full_sensor_ob
    assert isinstance(full_sensor_ob["obstacles"], dict)
    assert isinstance(full_sensor_ob["obstacles"][2], dict)
    assert isinstance(full_sensor_ob["obstacles"][2]["position"], np.ndarray)
    assert isinstance(full_sensor_ob["obstacles"][3], dict)
    assert isinstance(full_sensor_ob["obstacles"][3]["position"], np.ndarray)
    assert isinstance(full_sensor_ob["goals"], dict)
    assert isinstance(full_sensor_ob["goals"][5]["position"], np.ndarray)
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]["velocity"],
        sphereObst1.velocity(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]["position"],
        sphereObst1.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][5]["position"],
        goal1.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][5]["is_primary_goal"],
        goal1.is_primary_goal(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]["size"],
        sphereObst1.radius(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][3]["velocity"],
        dynamicSphereObst3.velocity(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][3]["position"],
        dynamicSphereObst3.position(t=env.t()),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][3]["size"],
        dynamicSphereObst3.radius(),
        decimal=2,
    )
    # Estimate the position of the movable obstacle after 10 steps
    pos = movable_obstacle.position(t=0)
    vel = movable_obstacle.velocity(t=0)
    acc = np.array([0, 0, -9.81])
    for i in range(10):
        vel += acc * env.dt
        pos += vel * env.dt
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][4]["position"],
        pos,
        decimal=1,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][4]["velocity"],
        vel,
        decimal=1,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][4]["size"],
        movable_obstacle.size(),
        decimal=2,
    )
    env.close()


def test_full_sensor_mujoco():
    sensor = FullSensor(
        goal_mask=["position", "is_primary_goal"],
        obstacle_mask=["position", "size"],
        variance=0.0,
        physics_engine_name="mujoco",
    )
    robot_model = RobotModel("pointRobot", "pointRobot")

    xml_file = robot_model.get_xml_path()
    robots = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    env: GenericMujocoEnv = gym.make(
        "generic-mujoco-env-v0",
        robots=robots,
        obstacles=[sphereObst1, dynamicSphereObst3, movable_obstacle],
        goals=[goal1],
        sensors=[sensor],
        render=False,
        enforce_real_time=False,
    ).unwrapped
    action = np.random.random(env.nu)
    for _ in range(10):
        ob, *_ = env.step(action)
    full_sensor_ob = ob["robot_0"]["FullSensor"]
    assert "obstacles" in full_sensor_ob.keys()
    assert "goals" in full_sensor_ob
    assert isinstance(full_sensor_ob["obstacles"], dict)
    assert isinstance(full_sensor_ob["obstacles"][0], dict)
    assert isinstance(full_sensor_ob["obstacles"][0]["position"], np.ndarray)
    assert isinstance(full_sensor_ob["goals"], dict)
    assert isinstance(full_sensor_ob["goals"][0]["position"], np.ndarray)
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][0]["position"],
        sphereObst1.position(t=env.t),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][0]["position"],
        goal1.position(t=env.t),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["goals"][0]["is_primary_goal"],
        goal1.is_primary_goal(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][0]["size"],
        sphereObst1.radius(),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][1]["position"],
        dynamicSphereObst3.position(t=env.t),
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][1]["size"],
        dynamicSphereObst3.radius(),
        decimal=2,
    )
    # Estimate the position of the movable obstacle after 10 steps
    pos = movable_obstacle.position(t=0)
    vel = movable_obstacle.velocity(t=0)
    acc = np.array([0, 0, -9.81])
    for i in range(10):
        vel += acc * env.dt
        pos += vel * env.dt
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]["position"],
        pos,
        decimal=1,
    )
    np.testing.assert_array_almost_equal(
        full_sensor_ob["obstacles"][2]["size"],
        movable_obstacle.size(),
        decimal=2,
    )
    env.close()
