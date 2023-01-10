import gym
import numpy as np
import pytest

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine

@pytest.fixture
def pointRobotEnv():
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    urdf = "pointRobot.urdf"
    return (urdf, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    urdf="panda.urdf"
    return (urdf, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    n = 1
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    urdf=f"nlink_{n}.urdf"
    return (urdf, init_pos, init_vel)

@pytest.fixture
def dualArmEnv():
    urdf = "dual_arm.urdf"
    init_pos = np.zeros(5)
    init_vel = np.zeros(5)
    return (urdf, init_pos, init_vel)

@pytest.fixture
def boxerRobotEnv():
    from urdfenvs.robots.boxer import BoxerRobot
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    return (BoxerRobot, init_pos, init_vel)

@pytest.fixture
def jackal_robot_env():
    from urdfenvs.robots.jackal import JackalRobot
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    return (JackalRobot, init_pos, init_vel)

@pytest.fixture
def tiagoReacherEnv():
    from urdfenvs.robots.tiago import TiagoRobot
    init_pos = np.zeros(20)
    init_pos[3] = 0.1
    init_vel = np.zeros(19)
    return (TiagoRobot, init_pos, init_vel)

@pytest.fixture
def albertReacherEnv():
    from urdfenvs.robots.albert import AlbertRobot
    init_pos = np.zeros(10)
    init_pos[6] = -1.501
    init_pos[8] = 1.8675
    init_pos[9] = -np.pi/4
    init_vel = np.zeros(9)
    return (AlbertRobot, init_pos, init_vel)

@pytest.fixture
def priusEnv():
    from urdfenvs.robots.prius import Prius
    initPos = np.zeros(3)
    initVel = np.zeros(2)
    return (Prius, initPos, initVel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv, dualArmEnv):
    return list(locals().values())

@pytest.fixture
def allDifferentialDriveEnvs(boxerRobotEnv, jackal_robot_env, tiagoReacherEnv, albertReacherEnv):
    return list(locals().values())

@pytest.fixture
def allBicycleModelEnvs(priusEnv):
    return list(locals().values())

def test_all(allEnvs):
    for setup in allEnvs:
        physics_engine = BulletPhysicsEngine(False)
        robot = GenericUrdfReacher(physics_engine, urdf=setup[0], mode="acc")
        env = gym.make("urdf-env-v0", robots=[robot], dt=0.01)
        ob = env.reset(pos=setup[1], vel=setup[2])
        action = np.random.random(env.n())
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], setup[1], decimal=2)
        ob, _, _, _ = env.step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env.n()
        env.close()

def test_allDifferentialDrive(allDifferentialDriveEnvs):
    for setup in allDifferentialDriveEnvs:
        physics_engine = BulletPhysicsEngine(False)
        robot = setup[0](physics_engine, mode='acc')
        env = gym.make("urdf-env-v0", robots=[robot], dt=0.01)
        ob = env.reset(pos=setup[1], vel=setup[2])
        action = np.random.random(env.n()) * 0.1
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], setup[1], decimal=2)
        ob, _, _, _ = env.step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['forward_velocity'], np.ndarray)
        env.close()

def test_allBicycleModel(allBicycleModelEnvs):
    for setup in allBicycleModelEnvs:
        physics_engine = BulletPhysicsEngine(False)
        robot = setup[0](physics_engine, mode='vel')
        env = gym.make("urdf-env-v0", robots=[robot], dt=0.01)
        ob = env.reset(pos=setup[1], vel=setup[2])
        action = np.random.random(env.n()) * 0.1
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], setup[1], decimal=2)
        ob, _, _, _ = env.step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['forward_velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env.n() + 1
        assert ob['robot_0']['joint_state']['velocity'].size == env.n() + 1
        assert ob['robot_0']['joint_state']['forward_velocity'].size == 2
        env.close()
