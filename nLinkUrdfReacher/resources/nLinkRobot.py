import os
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class NLinkRobot(AbstractRobot):
    def __init__(self, n):
        fileName = os.path.join(os.path.dirname(__file__), 'nlink_' + str(n) + '.urdf')
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.robot_joints = list(range(1, self.n() + 1))
        self.urdf_joints = list(range(1, self.n() + 1))

    def setAccLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

