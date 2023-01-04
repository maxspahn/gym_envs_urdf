import os
import numpy as np
from urdfenvs.urdf_common.differential_drive_robot import DifferentialDriveRobot


class JackalRobot(DifferentialDriveRobot):
    def __init__(self, mode: str):
        n = 2
        urdf_file = os.path.join(os.path.dirname(__file__), "jackal.urdf")
        super().__init__(n, urdf_file, mode, number_actuated_axes=2)
        self._wheel_radius = 0.098
        self._wheel_distance = 2 * 0.187795 + 0.08

    def set_joint_names(self):
        wheel_joint_names = [
            "rear_right_wheel",
            "rear_left_wheel",
            "front_right_wheel",
            "front_left_wheel"
        ]
        self._joint_names = (
            wheel_joint_names
        )

    def set_acceleration_limits(self):
        acc_limit = np.array([1.0, 1.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

