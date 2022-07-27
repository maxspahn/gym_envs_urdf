from urdfenvs.n_link_urdf_reacher.resources.n_link_robot import NLinkRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class NLinkUrdfReacherEnv(UrdfEnv):
    def __init__(self, n=3, **kwargs):
        super().__init__(NLinkRobot(n), **kwargs)
