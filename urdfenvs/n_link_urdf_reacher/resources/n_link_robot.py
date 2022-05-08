import os
import numpy as np
from urdfenvs.urdfCommon.holonomic_robot import HolonomicRobot


class NLinkRobot(HolonomicRobot):
    def __init__(self, n):
        urdf_file = os.path.join(
            os.path.dirname(__file__), "nlink_" + str(n) + ".urdf"
        )
        super().__init__(n, urdf_file)

    def set_joint_names(self):
        self._robot_joints = list(range(1, self.n() + 1))
        self._urdf_joints = list(range(1, self.n() + 1))

    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n)
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit
