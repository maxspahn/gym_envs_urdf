import os
from urdfenvs.generic_urdf_reacher.envs.generic_urdf_reacher_env import GenericUrdfReacherEnv


class NLinkUrdfReacherEnv(GenericUrdfReacherEnv):
    def __init__(self, n=3, **kwargs):
        urdf_file = os.path.join(
            os.path.dirname(__file__), "../resources/nlink_" + str(n) + ".urdf"
        )
        super().__init__(urdf_file, **kwargs)
