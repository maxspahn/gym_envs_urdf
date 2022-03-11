import os
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class AlbertRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 9
        _urdfFile = os.path.join(os.path.dirname(__file__), 'albert.urdf')
        super().__init__(n, _urdfFile)
        self._wheelRadius = 0.08
        self._wheelDistance = 0.494

    def setJointIndices(self):
        self.urdf_joints = [10, 11, 14, 15, 16, 17, 18, 19, 20]
        self.robot_joints = [24, 25, 8, 9, 10, 11, 12, 13, 14]
        self.castor_joints = [22, 23]

    def setAccelerationLimits(self):
        accLimit = np.array([1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def correctBaseOrientation(self, posBase):
        posBase[2] -= np.pi / 2.0
        if posBase[2] < -np.pi:
            posBase[2] += 2 * np.pi
        return posBase
