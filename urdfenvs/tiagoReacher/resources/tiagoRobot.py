import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class TiagoRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 19
        self._r = 0.1
        self._l = 0.4044
        fileName = os.path.join(os.path.dirname(__file__), "tiago_dual.urdf")
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.robot_joints = [6, 8, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 37, 38, 39, 40, 41, 42, 43]
        self.urdf_joints = [6, 8, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 38, 39, 40, 41, 42, 43, 44]
        self.castor_joints = [9, 10, 11, 12, 13, 14, 15, 16]
        # TODO: This could be used as a starting point to get joint indices from urdf <01-12-21, mspahn> #
        """
        wheel_joint_names = ["wheel_right_joint", "wheel_left_joint"]
        torso_joint_name = ["torso_lift_joint"]
        head_joint_names = ["head_" + str(i) + "_joint" for i in range(3)]
        arm_right_joint_names = ["arm_right_" + str(i) + "_joint" for i in range(8)]
        arm_left_joint_names = ["arm_left_" + str(i) + "_joint" for i in range(8)]
        self._joint_names = (
            wheel_joint_names
            + torso_joint_name
            + head_joint_names
            + arm_right_joint_names
            + arm_left_joint_names
        )
        robot = URDF.load(self.fileName)
        self.urdf_joints = []
        for i, joint in enumerate(robot.joints):
            if joint.name in self._joint_names:
                self.urdf_joints.append(i)

        self.robot_joints = []
        self.caster_joints = []
        for _id in range(p.getNumJoints(self.robot)):
            joint_name = p.getJointInfo(self.robot, _id)[1].decode("UTF-8")
            if joint_name in self._joint_names:
                self.robot_joints.append(_id)
            if "caster" in joint_name:
                self.caster_joints.append(_id)
        __import__('pdb').set_trace()
        self.robot_joints_gripper = []
        """

    def setAccLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

