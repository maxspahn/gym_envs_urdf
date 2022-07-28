from urdfenvs.dual_arm.envs.dual_arm_env import DualArmEnv
from urdfenvs.generic_urdf_reacher.envs.acc import GenericUrdfReacherAccEnv

class DualArmAccEnv(DualArmEnv, GenericUrdfReacherAccEnv):
    pass
