import numpy as np
from pandaReacher.resources.pandaRobot import PandaRobot
from urdfCommon.urdfEnv import UrdfEnv


class PandaReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, friction=0.0, gripper=False, n=7):
        super().__init__(PandaRobot(gripper=gripper, friction=friction), render=render, dt=dt)
        self.setSpaces()

    def checkInitialState(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.robot.n():
            pos = np.zeros(self.robot.n())
            pos[3] = -1.501
            pos[5] = 1.8675
            pos[6] = np.pi/4
        if not isinstance(vel, np.ndarray) or not vel.size == self.robot.n():
            vel = np.zeros(self.robot.n())
        return pos, vel
