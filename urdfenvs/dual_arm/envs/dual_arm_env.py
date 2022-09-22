import os

from urdfenvs.generic_urdf_reacher.envs.generic_urdf_reacher_env import GenericUrdfReacherEnv


class DualArmEnv(GenericUrdfReacherEnv):
    def __init__(self, **kwargs):
        urdf_file = os.path.join(
            os.path.dirname(__file__), "../resources/dual_arm.urdf"
        )
        super().__init__(urdf_file, **kwargs)
