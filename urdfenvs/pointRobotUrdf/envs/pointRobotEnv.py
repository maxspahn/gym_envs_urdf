from urdfenvs.pointRobotUrdf.resources.pointRobot import PointRobot
from urdfenvs.urdfCommon.urdfEnv import UrdfEnv


class PointRobotEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01):
        super().__init__(PointRobot(), render=render, dt=dt)
        self.setSpaces()
        self.reset(initialSet=True)
