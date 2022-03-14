import gym
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    import urdfenvs.point_robot_urdf
    initPos = np.array([0.0, -1.0, 0.0])
    initVel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-vel-v0", render=False, dt=0.01)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def pandaRobotEnv():
    import urdfenvs.panda_reacher
    initPos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    initVel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    env = gym.make("panda-reacher-vel-v0", dt=0.01, render=False, gripper=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def nLinkRobotEnv():
    n = 1
    import urdfenvs.n_link_urdf_reacher
    initPos = np.array([0.0])
    initVel = np.array([0.0])
    env = gym.make("nLink-urdf-reacher-vel-v0", n=n, dt=0.01, render=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def boxerRobotEnv():
    import urdfenvs.boxer_robot
    initPos = np.array([0.0, 0.0, 0.0])
    initVel = np.array([0.0, 0.0])
    env = gym.make("boxer-robot-vel-v0", dt=0.01, render=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def tiagoReacherEnv():
    import urdfenvs.tiago_reacher
    initPos = np.zeros(20)
    initPos[3] = 0.1
    initVel = np.zeros(19)
    env = gym.make("tiago-reacher-vel-v0", dt=0.01, render=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def albertReacherEnv():
    import urdfenvs.albert_reacher
    initPos = np.zeros(10)
    initPos[6] = -1.501
    initPos[8] = 1.8675
    initPos[9] = -np.pi/4
    initVel = np.zeros(9)
    env = gym.make("albert-reacher-vel-v0", dt=0.01, render=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv):
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

