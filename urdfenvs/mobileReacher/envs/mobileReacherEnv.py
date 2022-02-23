from abc import abstractmethod
import numpy as np

from urdfenvs.mobileReacher.resources.mobileRobot import MobileRobot
from urdfenvs.urdfCommon.urdfEnv import UrdfEnv


class MobileReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, gripper=False):
        super().__init__(MobileRobot(gripper=gripper), render=render, dt=dt)
        self._n = self.robot.n()
        self.setSpaces()

    def checkInitialState(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.robot.n():
            pos = np.zeros(self.robot.n())
            pos[6] = -1.501
            pos[8] = 1.8675
            pos[9] = np.pi/4
            if self.robot.n() > 10:
                pos[10] = 0.02
                pos[11] = 0.02
        if not isinstance(vel, np.ndarray) or not vel.size == self.robot.n():
            vel = np.zeros(self.robot.n())
        return pos, vel

    @abstractmethod
    def setSpaces(self):
        pass

    @abstractmethod
    def applyAction(self, action):
        pass

