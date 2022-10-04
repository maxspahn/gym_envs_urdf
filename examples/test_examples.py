import pytest
import warnings

from albert import run_albert
from boxer import run_boxer
from dual_arm import run_dual_arm
from generic_holonomic import run_generic_holonomic
from iris import run_iris
from panda_reacher import run_panda
from mobile_reacher import run_mobile_reacher
from n_link_urdf_reacher import run_n_link_reacher
from prius import run_prius
from point_robot import run_point_robot



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
    blueprint_test(run_boxer)

def test_albert_robot():
    blueprint_test(run_albert)

def test_dual_arm_robot():
    blueprint_test(run_dual_arm)

def test_generic_holonomic_robot():
    blueprint_test(run_generic_holonomic)

def test_iris_robot():
    blueprint_test(run_iris)

def test_panda_robot():
    blueprint_test(run_panda)

def test_mobile_reacher_robot():
    blueprint_test(run_mobile_reacher)

def test_n_link_reacher_robot():
    blueprint_test(run_n_link_reacher)

def test_prius_robot():
    blueprint_test(run_prius)

def test_point_robot_robot():
    blueprint_test(run_point_robot)
