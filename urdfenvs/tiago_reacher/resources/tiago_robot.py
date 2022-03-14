import os
import numpy as np

from urdfenvs.urdfCommon.differential_drive_robot import DifferentialDriveRobot


class TiagoRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 19
        urdf_file = os.path.join(os.path.dirname(__file__), "tiago_dual.urdf")
        super().__init__(n, urdf_file)
        self._wheel_radius = 0.1
        self._wheel_distance = 0.4044
        self._spawn_offset = np.array([-0.1764081, 0.0, 0.1])

    def set_joint_indices(self):
        self._robot_joints = [
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
        self._urdf_joints = [
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
        self._castor_joints = [9, 10, 11, 12, 13, 14, 15, 16]

    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n)
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit
