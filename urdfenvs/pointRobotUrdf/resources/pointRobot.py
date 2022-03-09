import os
import numpy as np

from urdfenvs.urdfCommon.holonomicRobot import HolonomicRobot


class PointRobot(HolonomicRobot):
    def __init__(self):
        _urdfFile = os.path.join(os.path.dirname(__file__), 'pointRobot.urdf')
        n = 3
        super().__init__(n, _urdfFile)

    def setJointIndices(self):
        self.robot_joints = [0, 1, 2]
        self.urdf_joints = [0, 1, 2]

    def setAccelerationLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

