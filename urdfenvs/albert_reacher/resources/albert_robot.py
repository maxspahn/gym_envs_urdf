import os
import numpy as np

from urdfenvs.urdfCommon.differential_drive_robot import DifferentialDriveRobot


class AlbertRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 9
        urdf_file = os.path.join(os.path.dirname(__file__), "albert.urdf")
        super().__init__(n, urdf_file)
        self._wheel_radius = 0.08
        self._wheel_distance = 0.494

    def set_joint_indices(self):
        self._urdf_joints = [10, 11, 14, 15, 16, 17, 18, 19, 20]
        self._robot_joints = [24, 25, 8, 9, 10, 11, 12, 13, 14]
        self._castor_joints = [22, 23]

    def set_acceleration_limits(self):
        acc_limit = np.array(
            [1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
        )
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

    def correct_base_orientation(self, pos_base):
        pos_base[2] -= np.pi / 2.0
        if pos_base[2] < -np.pi:
            pos_base[2] += 2 * np.pi
        return pos_base
