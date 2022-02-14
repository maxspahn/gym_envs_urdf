import gym
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    import urdfenvs.pointRobotUrdf
    initPos = np.array([0.0, -1.0, 0.0])
    initVel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-acc-v0", render=False, dt=0.01)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def pandaRobotEnv():
    import urdfenvs.pandaReacher
    initPos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    initVel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    env = gym.make("panda-reacher-acc-v0", dt=0.01, render=False, gripper=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def nLinkRobotEnv():
    n = 1
    import urdfenvs.nLinkUrdfReacher
    initPos = np.array([0.0])
    initVel = np.array([0.0])
    env = gym.make("nLink-urdf-reacher-acc-v0", n=n, dt=0.01, render=False)
    ob = env.reset(pos=initPos, vel=initVel)
    return (env, initPos, initVel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv):
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

