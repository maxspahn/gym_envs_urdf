import gym
import numpy as np
import pytest
import urdfenvs.urdf_common

@pytest.fixture
def pointRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    robot = GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    robot = GenericUrdfReacher(urdf="panda.urdf", mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    n = 1
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    robot = GenericUrdfReacher(urdf=f"nlink_{n}.urdf", mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def boxerRobotEnv():
    from urdfenvs.robots.boxer import BoxerRobot
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    robot = BoxerRobot(mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def tiagoReacherEnv():
    from urdfenvs.robots.tiago import TiagoRobot
    init_pos = np.zeros(20)
    init_pos[3] = 0.1
    init_vel = np.zeros(19)
    robot = TiagoRobot(mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def albertReacherEnv():
    from urdfenvs.robots.albert import AlbertRobot
    init_pos = np.zeros(10)
    init_pos[6] = -1.501
    init_pos[8] = 1.8675
    init_pos[9] = -np.pi/4
    init_vel = np.zeros(9)
    robot = AlbertRobot(mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def priusEnv():
    from urdfenvs.robots.prius import Prius
    initPos = np.zeros(3)
    initVel = np.zeros(2)
    robot = Prius(mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    _ = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def dualArmEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    robot = GenericUrdfReacher(urdf="dual_arm.urdf", mode="vel")
    env = gym.make("urdf-env-v0", robots=[robot], render=False, dt=0.01)
    init_pos = np.zeros(env.n())
    init_vel = np.zeros(env.n())
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv, dualArmEnv):
    return list(locals().values())

@pytest.fixture
def allDifferentialDriveEnvs(boxerRobotEnv, tiagoReacherEnv, albertReacherEnv):
    return list(locals().values())

@pytest.fixture
def allBicycleModelEnvs(priusEnv):
    return list(locals().values())

def test_all(allEnvs):
    for env in allEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n())
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env[0].n()
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['velocity'], action, decimal=2)

def test_allDifferentialDrive(allDifferentialDriveEnvs):
    for env in allDifferentialDriveEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n()) * 0.1
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['forward_velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env[0].n() + 1
        assert ob['robot_0']['joint_state']['velocity'].size == env[0].n() + 1
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['forward_velocity'], action[0], decimal=2)
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['velocity'][3:], action[2:], decimal=2)

def test_allBicycleModel(allBicycleModelEnvs):
    for env in allBicycleModelEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n()) * 0.1
        np.testing.assert_array_almost_equal(ob['robot_0']['x'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['x'], np.ndarray)
        assert isinstance(ob['robot_0']['xdot'], np.ndarray)
        assert isinstance(ob['robot_0']['vel'], np.ndarray)
        assert ob['robot_0']['x'].size == env[0].n() + 1
        assert ob['robot_0']['xdot'].size == env[0].n() + 1
        assert ob['robot_0']['vel'].size == 2
        np.testing.assert_array_almost_equal(ob['robot_0']['vel'][0:1], action[0:1], decimal=2)
        np.testing.assert_array_almost_equal(ob['robot_0']['xdot'][3:], action[2:], decimal=2)

