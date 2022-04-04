from urdfenvs.dual_arm.resources.dual_arm_robot import DualArmRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class DualArmEnv(UrdfEnv):
    def __init__(self, render=False, dt=0.01, n=5):
        super().__init__(DualArmRobot(), render=render, dt=dt)
        self.set_spaces()
