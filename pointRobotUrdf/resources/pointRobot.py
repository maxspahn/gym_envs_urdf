import os
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class PointRobot(AbstractRobot):
    def __init__(self):
        fileName = os.path.join(os.path.dirname(__file__), 'pointRobot.urdf')
        n = 2
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.robot_joints = [0, 1]
        self.urdf_joints = [0, 1]

    def setAccLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

