import gym
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    robot = GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    robot = GenericUrdfReacher(urdf="panda.urdf", mode="acc")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    robot = GenericUrdfReacher(urdf="nlink_1.urdf", mode="acc")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def dualArmEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    robot = GenericUrdfReacher(urdf="dual_arm.urdf", mode="acc")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    init_pos = np.zeros(env.n())
    init_vel = np.zeros(env.n())
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv, dualArmEnv):
    return list(locals().values())

def test_all(allEnvs):
    for env in allEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n())
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env[0].n()

