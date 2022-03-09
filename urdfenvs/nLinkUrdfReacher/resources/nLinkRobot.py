import os
import numpy as np

from urdfenvs.urdfCommon.holonomicRobot import HolonomicRobot


class NLinkRobot(HolonomicRobot):
    def __init__(self, n):
        _urdfFile = os.path.join(os.path.dirname(__file__), 'nlink_' + str(n) + '.urdf')
        super().__init__(n, _urdfFile)

    def setJointIndices(self):
        self.robot_joints = list(range(1, self.n() + 1))
        self.urdf_joints = list(range(1, self.n() + 1))

    def setAccelerationLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

