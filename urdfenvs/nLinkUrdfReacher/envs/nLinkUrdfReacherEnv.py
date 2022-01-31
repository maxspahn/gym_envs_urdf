from nLinkUrdfReacher.resources.nLinkRobot import NLinkRobot
from urdfCommon.urdfEnv import UrdfEnv


class NLinkUrdfReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, n=3):
        super().__init__(NLinkRobot(n), render=render, dt=dt)
        self.setSpaces()
