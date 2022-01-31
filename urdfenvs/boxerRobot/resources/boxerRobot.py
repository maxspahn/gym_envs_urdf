import pybullet as p
import os
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class BoxerRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 2
        fileName = os.path.join(os.path.dirname(__file__), 'boxer.urdf')
        self._r = 0.08
        self._l = 0.494
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.urdf_joints = [2, 3]
        self.robot_joints = [3, 4]
        self.castor_joints = [1, 2]

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit
