import os
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class BoxerRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 2
        fileName = os.path.join(os.path.dirname(__file__), 'boxer.urdf')
        super().__init__(n, fileName)
        self._r = 0.08
        self._l = 0.494

    def setJointIndices(self):
        self.urdf_joints = [2, 3]
        self.robot_joints = [4, 5]
        self.castor_joints = [2, 3]

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit
