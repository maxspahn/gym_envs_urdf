import gym
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    robot = GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc")
    return (robot, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    robot = GenericUrdfReacher(urdf="panda.urdf", mode="acc")
    return (robot, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    n = 1
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    robot = GenericUrdfReacher(urdf=f"nlink_{n}.urdf", mode="acc")
    return (robot, init_pos, init_vel)

@pytest.fixture
def dualArmEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    robot = GenericUrdfReacher(urdf="dual_arm.urdf", mode="acc")
    init_pos = np.zeros(robot.n())
    init_vel = np.zeros(robot.n())
    return (robot, init_pos, init_vel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv, dualArmEnv):
    return list(locals().values())

def test_all(allEnvs):
    for setup in allEnvs:
        env = gym.make("urdf-env-v0", robots=[setup[0]], render=False, dt=0.01)
        ob = env.reset(pos=setup[1], vel=setup[2])
        action = np.random.random(env.n())
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], setup[1], decimal=2)
        ob, _, _, _ = env.step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env.n()
        env.close()
