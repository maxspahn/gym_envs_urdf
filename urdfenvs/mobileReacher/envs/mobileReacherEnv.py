from abc import abstractmethod

from urdfenvs.mobileReacher.resources.mobileRobot import MobileRobot
from urdfenvs.urdfCommon.urdfEnv import UrdfEnv


class MobileReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, gripper=False):
        super().__init__(MobileRobot(gripper=gripper), render=render, dt=dt)
        self._n = self.robot.n()
        self.setSpaces()

    @abstractmethod
    def setSpaces(self):
        pass

    @abstractmethod
    def applyAction(self, action):
        pass

