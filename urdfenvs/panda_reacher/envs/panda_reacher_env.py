import os

from urdfenvs.generic_urdf_reacher.envs.generic_urdf_reacher_env import GenericUrdfReacherEnv

class PandaReacherEnv(GenericUrdfReacherEnv):
    def __init__(self, friction=0.0, gripper=False, **kwargs):
        if gripper:
            urdf_file = os.path.join(
                os.path.dirname(__file__), "../resources/panda_with_gripper.urdf"
            )
        else:
            urdf_file = os.path.join(
                os.path.dirname(__file__), "../resources/panda.urdf"
            )
        super().__init__(urdf_file, **kwargs)

