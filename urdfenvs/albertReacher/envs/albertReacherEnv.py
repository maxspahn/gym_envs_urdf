import numpy as np

from urdfenvs.albertReacher.resources.albertRobot import AlbertRobot
from urdfenvs.urdfCommon.urdfEnv import UrdfEnv


class AlbertReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01):
        super().__init__(AlbertRobot(), render=render, dt=dt)
        self.setSpaces()

    def checkInitialState(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.robot.n()+1:
            pos = np.zeros(self.robot.n()+1)
            pos[6] = -1.501
            pos[8] = 1.8675
            pos[9] = np.pi/4
        if not isinstance(vel, np.ndarray) or not vel.size == self.robot.n():
            vel = np.zeros(self.robot.n())
        return pos, vel
