import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np

from urdfCommon.diffDriveRobot import DiffDriveRobot


class AlbertRobot(DiffDriveRobot):
    def __init__(self):
        n = 9
        fileName = os.path.join(os.path.dirname(__file__), 'albert.urdf')
        self._r = 0.08
        self._l = 0.494
        super().__init__(n, fileName)


    def setJointIndices(self):
        self.urdf_joints = [10, 11, 14, 15, 16, 17, 18, 19, 20]
        self.robot_joints = [23, 24, 7, 8, 9, 10, 11, 12, 13]
        self.castor_joints = [21, 22]

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

