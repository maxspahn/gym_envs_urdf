import os
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class TiagoRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 19
        _urdfFile = os.path.join(os.path.dirname(__file__), "tiago_dual.urdf")
        super().__init__(n, _urdfFile)
        self._r = 0.1
        self._l = 0.4044

    def setJointIndices(self):
        self.robot_joints = [
            6,
            8,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
        ]
        self.urdf_joints = [
            6,
            8,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ]
        self.castor_joints = [9, 10, 11, 12, 13, 14, 15, 16]

    def setAccelerationLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit
