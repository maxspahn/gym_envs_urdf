import gym
import numpy as np
import pytest
import urdfenvs.albert_reacher
import urdfenvs.point_robot_urdf
import urdfenvs.panda_reacher
import urdfenvs.n_link_urdf_reacher
import urdfenvs.dual_arm
import urdfenvs.tiago_reacher
import urdfenvs.boxer_robot

@pytest.fixture
def pointRobotEnv():
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-vel-v0", render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    env = gym.make("panda-reacher-vel-v0", dt=0.01, render=False, gripper=False)
    ob = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    n = 1
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    env = gym.make("nLink-urdf-reacher-vel-v0", n=n, dt=0.01, render=False)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def boxerRobotEnv():
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    env = gym.make("boxer-robot-vel-v0", dt=0.01, render=False)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def tiagoReacherEnv():
    init_pos = np.zeros(20)
    init_pos[3] = 0.1
    init_vel = np.zeros(19)
    env = gym.make("tiago-reacher-vel-v0", dt=0.01, render=False)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def albertReacherEnv():
    init_pos = np.zeros(10)
    init_pos[6] = -1.501
    init_pos[8] = 1.8675
    init_pos[9] = -np.pi/4
    init_vel = np.zeros(9)
    env = gym.make("albert-reacher-vel-v0", dt=0.01, render=False)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def dualArmEnv():
    env = gym.make("dual-arm-vel-v0", dt=0.01, render=False)
    init_pos = np.zeros(env.n())
    init_vel = np.zeros(env.n())
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv, dualArmEnv):
    return list(locals().values())

@pytest.fixture
def allNonHolonomicEnvs(boxerRobotEnv, tiagoReacherEnv, albertReacherEnv):
    return list(locals().values())

def test_all(allEnvs):
    for env in allEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n())
        np.testing.assert_array_almost_equal(ob['x'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['x'], np.ndarray)
        assert isinstance(ob['xdot'], np.ndarray)
        assert ob['x'].size == env[0].n()
        np.testing.assert_array_almost_equal(ob['xdot'], action, decimal=2)

def test_allNonHolonomic(allNonHolonomicEnvs):
    for env in allNonHolonomicEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n()) * 0.1
        np.testing.assert_array_almost_equal(ob['x'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['x'], np.ndarray)
        assert isinstance(ob['xdot'], np.ndarray)
        assert isinstance(ob['vel'], np.ndarray)
        assert ob['x'].size == env[0].n() + 1
        assert ob['xdot'].size == env[0].n() + 1
        assert ob['vel'].size == 2
        np.testing.assert_array_almost_equal(ob['vel'], action[0:2], decimal=2)
        np.testing.assert_array_almost_equal(ob['xdot'][3:], action[2:], decimal=2)

