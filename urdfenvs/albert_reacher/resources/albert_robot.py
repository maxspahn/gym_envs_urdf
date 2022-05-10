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

    def set_joint_names(self):
        wheel_joint_names = ["wheel_right_joint", "wheel_left_joint"]
        mmrobot_joint_name = ["mmrobot_joint" + str(i) for i in range(1,8)]
        self._joint_names = (
            wheel_joint_names
            + mmrobot_joint_name
        )

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
