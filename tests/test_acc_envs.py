import gym
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    import urdfenvs.point_robot_urdf
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-acc-v0", render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    import urdfenvs.panda_reacher
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    env = gym.make("panda-reacher-acc-v0", dt=0.01, render=False, gripper=False)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    n = 1
    import urdfenvs.n_link_urdf_reacher
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    env = gym.make("nLink-urdf-reacher-acc-v0", n=n, dt=0.01, render=False)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return (env, init_pos, init_vel)

@pytest.fixture
def dualArmEnv():
    import urdfenvs.dual_arm
    env = gym.make("dual-arm-acc-v0", dt=0.01, render=False)
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
        np.testing.assert_array_almost_equal(ob['x'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['x'], np.ndarray)
        assert isinstance(ob['xdot'], np.ndarray)
        assert ob['x'].size == env[0].n()

