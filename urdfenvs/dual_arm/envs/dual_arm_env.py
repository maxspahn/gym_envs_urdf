from urdfenvs.dual_arm.resources.dual_arm_robot import DualArmRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class DualArmEnv(UrdfEnv):
    def __init__(self, **kwargs):
        super().__init__(DualArmRobot(), **kwargs)
