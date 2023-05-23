import gym
import numpy as np
import pytest
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.robots.generic_urdf import GenericDiffDriveRobot

@pytest.fixture
def pointRobotEnv():
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    robot = GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel")
    return (robot, init_pos, init_vel)

@pytest.fixture
def pandaRobotEnv():
    init_pos = np.array([0.0, 0.0, 0.0, -1.875, 0.0, 1.5, 0.0])
    init_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0])
    robot = GenericUrdfReacher(urdf="panda.urdf", mode="vel")
    return (robot, init_pos, init_vel)

@pytest.fixture
def nLinkRobotEnv():
    n = 1
    init_pos = np.array([0.0])
    init_vel = np.array([0.0])
    robot = GenericUrdfReacher(urdf=f"nlink_{n}.urdf", mode="vel")
    return (robot, init_pos, init_vel)

@pytest.fixture
def boxerRobotEnv():
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    robot = GenericDiffDriveRobot(
        urdf="boxer.urdf",
        mode="vel",
        actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
        castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
        wheel_radius = 0.08,
        wheel_distance = 0.494,
    )
    return (robot, init_pos, init_vel)

@pytest.fixture
def jackal_robot_env():
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0])
    robot = GenericDiffDriveRobot(
        urdf="jackal.urdf",
        mode="vel",
        actuated_wheels=[
            "rear_right_wheel",
            "rear_left_wheel",
            "front_right_wheel",
            "front_left_wheel",
        ],
        castor_wheels=[],
        wheel_radius = 0.098,
        wheel_distance = 2 * 0.187795 + 0.08,
    )
    return (robot, init_pos, init_vel)

@pytest.fixture
def tiagoReacherEnv():
    init_pos = np.zeros(24)
    init_pos[3] = 0.1
    init_vel = np.zeros(23)
    robot = GenericDiffDriveRobot(
        urdf="tiago_dual.urdf",
        mode="vel",
        actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
        castor_wheels=[
            "caster_front_right_2_joint",
            "caster_front_left_2_joint",
            "caster_back_right_2_joint",
            "caster_back_left_2_joint",
        ],
        not_actuated_joints=[
            "suspension_right_joint",
            "suspension_left_joint",
        ],
        wheel_radius = 0.1,
        wheel_distance = 0.4044,
        spawn_offset = np.array([-0.1764081, 0.0, 0.1]),
    )
    return (robot, init_pos, init_vel)

@pytest.fixture
def albertReacherEnv():
    init_pos = np.zeros(12)
    init_pos[6] = -1.501
    init_pos[8] = 1.8675
    init_pos[9] = -np.pi/4
    init_vel = np.zeros(11)
    robot = GenericDiffDriveRobot(
        urdf="albert.urdf",
        mode="vel",
        actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
        castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
        wheel_radius = 0.08,
        wheel_distance = 0.494,
    )
    return (robot, init_pos, init_vel)

@pytest.fixture
def priusEnv():
    from urdfenvs.robots.prius import Prius
    initPos = np.zeros(3)
    initVel = np.zeros(2)
    robot = Prius(mode="vel")
    return (robot, initPos, initVel)

@pytest.fixture
def dualArmEnv():
    from urdfenvs.robots.generic_urdf import GenericUrdfReacher
    robot = GenericUrdfReacher(urdf="dual_arm.urdf", mode="vel")
    init_pos = np.zeros(robot.n())
    init_vel = np.zeros(robot.n())
    return (robot, init_pos, init_vel)

@pytest.fixture
def allEnvs(pointRobotEnv, pandaRobotEnv, nLinkRobotEnv, dualArmEnv):
    return list(locals().values())

@pytest.fixture
def allDifferentialDriveEnvs(boxerRobotEnv, jackal_robot_env, tiagoReacherEnv, albertReacherEnv):
    return list(locals().values())

@pytest.fixture
def allBicycleModelEnvs(priusEnv):
    return list(locals().values())

def test_all_generic(allEnvs):
    for setup in allEnvs:
        env = gym.make("urdf-env-v0", robots=[setup[0]], render=False, dt=0.01)
        ob, info= env.reset(pos=setup[1], vel=setup[2])
        action = np.random.random(env.n())
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], setup[1], decimal=2)
        ob, _, _, _ = env.step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env.n()
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['velocity'], action, decimal=2)
        env.close()

def test_allDifferentialDrive(allDifferentialDriveEnvs):
    for setup in allDifferentialDriveEnvs:
        env = gym.make("urdf-env-v0", robots=[setup[0]], render=False, dt=0.01)
        ob, info= env.reset(pos=setup[1], vel=setup[2])
        action = np.random.random(env.n()) * 0.02
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['position'], setup[1], decimal=2)
        ob, _, _, _ = env.step(action)
        assert isinstance(ob['robot_0'], dict)
        assert isinstance(ob['robot_0']['joint_state']['position'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['velocity'], np.ndarray)
        assert isinstance(ob['robot_0']['joint_state']['forward_velocity'], np.ndarray)
        assert ob['robot_0']['joint_state']['position'].size == env.n() + 1
        assert ob['robot_0']['joint_state']['velocity'].size == env.n() + 1
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['forward_velocity'], action[0], decimal=2)
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['velocity'][3:], action[2:], decimal=2)
        env.close()

def test_allBicycleModel(allBicycleModelEnvs):
    for setup in allBicycleModelEnvs:
        env = gym.make("urdf-env-v0", robots=[setup[0]], render=False, dt=0.01)
        ob, info= env.reset(pos=setup[1], vel=setup[2])
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
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['forward_velocity'][0:1], action[0:1], decimal=2)
        np.testing.assert_array_almost_equal(ob['robot_0']['joint_state']['velocity'][3:], action[2:], decimal=2)
        env.close()
