import os
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class BoxerRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 2
        _urdfFile = os.path.join(os.path.dirname(__file__), 'boxer.urdf')
        super().__init__(n, _urdfFile)
        self._wheelRadius = 0.08
        self._wheelDistance = 0.494

    def setJointIndices(self):
        self.urdf_joints = [2, 3]
        self.robot_joints = [4, 5]
        self.castor_joints = [2, 3]

    def setAccelerationLimits(self):
        accLimit = np.array([1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def correctBaseOrientation(self, posBase):
        posBase[2] -= np.pi / 2.0
        if posBase[2] < -np.pi:
            posBase[2] += 2 * np.pi
        return posBase
