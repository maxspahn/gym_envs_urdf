from urdfenvs.n_link_urdf_reacher.resources.n_link_robot import NLinkRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class NLinkUrdfReacherEnv(UrdfEnv):
    def __init__(self, render=False, dt=0.01, n=3):
        super().__init__(NLinkRobot(n), render=render, dt=dt)
        self.set_spaces()
