import numpy as np

from urdfenvs.tiagoReacher.resources.tiagoRobot import TiagoRobot
from urdfenvs.urdfCommon.urdfEnv import UrdfEnv


class TiagoReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, n=19):
        super().__init__(TiagoRobot(), render=render, dt=dt)
        self.setSpaces()

    def checkInitialState(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.robot.n() + 1:
            pos = np.zeros(self.robot.n()+1)
            pos[3] = 0.1
        if not isinstance(vel, np.ndarray) or not vel.size == self.robot.n():
            vel = np.zeros(self.robot.n())
        return pos, vel
