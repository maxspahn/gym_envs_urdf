import pytest
import warnings

def blueprint_test(test_main):
    """
    Blueprint for environment tests.
    An environment main always has the four arguments:
        - n_steps: int
        - render: bool
        - goal: bool
        - obstacles: bool

    The function verifies if the main returns a list of observations.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        history = test_main(n_steps=100, render=False, goal=True, obstacles=True)
    assert isinstance(history, list)


def test_boxer_robot():
    from boxer import run_boxer
    blueprint_test(run_boxer)

def test_jackal_robot():
    from jackal import run_jackal
    blueprint_test(run_jackal)

def test_albert_robot():
    from albert import run_albert
    blueprint_test(run_albert)

def test_dual_arm_robot():
    from dual_arm import run_dual_arm
    blueprint_test(run_dual_arm)

def test_generic_holonomic_robot():
    from generic_holonomic import run_generic_holonomic
    blueprint_test(run_generic_holonomic)

def test_iris_robot():
    from iris import run_iris
    blueprint_test(run_iris)

def test_panda_robot():
    from panda_reacher import run_panda
    blueprint_test(run_panda)

def test_mobile_reacher_robot():
    from mobile_reacher import run_mobile_reacher
    blueprint_test(run_mobile_reacher)

def test_n_link_reacher_robot():
    from n_link_urdf_reacher import run_n_link_reacher
    blueprint_test(run_n_link_reacher)

def test_prius_robot():
    from prius import run_prius
    blueprint_test(run_prius)

def test_point_robot():
    from point_robot import run_point_robot
    blueprint_test(run_point_robot)

def test_point_robot_with_lidar():
    from point_robot_lidar_sensor import run_point_robot_with_lidar
    blueprint_test(run_point_robot_with_lidar)

def test_point_robot_with_obstacle_sensor():
    from point_robot_obstacle_sensor import run_point_robot_with_obstacle_sensor
    blueprint_test(run_point_robot_with_obstacle_sensor)

def test_tiago_robot():
    from tiago import run_tiago
    blueprint_test(run_tiago)

def test_multi_robot():
    from multi_robot import run_multi_robot
    blueprint_test(run_multi_robot)

