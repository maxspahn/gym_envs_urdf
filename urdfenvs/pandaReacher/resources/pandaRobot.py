import os
import numpy as np

from urdfenvs.urdfCommon.holonomicRobot import HolonomicRobot


class PandaRobot(HolonomicRobot):
    def __init__(self, gripper=False, friction=0.0):
        self._gripper = gripper
        self._friction = friction
        if gripper:
            _urdfFile = os.path.join(
                os.path.dirname(__file__), "pandaWithGripper_working.urdf"
            )
            n = 9
        else:
            _urdfFile = os.path.join(os.path.dirname(__file__), "panda_working.urdf")
            n = 7
        super().__init__(n, _urdfFile)

    def setJointIndices(self):
        if self._gripper:
            self.robot_joints = [1, 2, 3, 4, 5, 6, 7, 9, 10]
            self.urdf_joints = [1, 2, 3, 4, 5, 6, 7, 9, 10]
        else:
            self.robot_joints = [1, 2, 3, 4, 5, 6, 7]
            self.urdf_joints = [1, 2, 3, 4, 5, 6, 7]

    def setAccelerationLimits(self):
        accLimit = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit[0: self.n()]
        self._limitAcc_j[1, :] = accLimit[0: self.n()]

