import os
import numpy as np
from urdfenvs.urdf_common.differential_drive_robot import DifferentialDriveRobot


class TiagoRobot(DifferentialDriveRobot):
    def __init__(self, mode: str):
        n = 19
        urdf_file = os.path.join(os.path.dirname(__file__), "tiago_dual.urdf")
        super().__init__(n, urdf_file, mode)
        self._wheel_radius = 0.1
        self._wheel_distance = 0.4044
        self._spawn_offset = np.array([-0.1764081, 0.0, 0.1])

    def set_joint_names(self):
        wheel_joint_names = ["wheel_right_joint", "wheel_left_joint"]
        torso_joint_name = ["torso_lift_joint"]
        head_joint_names = ["head_" + str(i) + "_joint" for i in range(3)]
        arm_right_joint_names = ["arm_right_" + str(i) +
                                    "_joint" for i in range(8)]
        arm_left_joint_names = ["arm_left_" + str(i) +
                                    "_joint" for i in range(8)]
        self._joint_names = (
            wheel_joint_names
            + torso_joint_name
            + head_joint_names
            + arm_left_joint_names
            + arm_right_joint_names
        )


    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n)
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

    def check_state(self, pos, vel):
        if (
            not isinstance(pos, np.ndarray)
            or not pos.size == self.n() + 1
        ):
            pos = np.zeros(self.n() + 1)
            pos[3] = 0.1
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel
