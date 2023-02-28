from typing import List
import numpy as np
from urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
from urdfenvs.urdf_common.differential_drive_robot import DifferentialDriveRobot

class GenericDiffDriveRobot(DifferentialDriveRobot):
    def __init__(self,
                 urdf: str,
                 mode: str,
                 actuated_wheels: List[str],
                 actuated_joints: List[str],
                 wheel_radius: float,
                 wheel_distance: float,
                 spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15]),
        ):
        number_actuated_axes = int(len(actuated_wheels)/2)
        n = 2 + len(actuated_joints)
        self._actuated_wheels = actuated_wheels
        self._actuated_joints = actuated_joints
        self._wheel_radius = wheel_radius
        self._wheel_distance = wheel_distance
        super().__init__(
            n,
            urdf,
            mode=mode,
            number_actuated_axes=number_actuated_axes,
            spawn_offset=spawn_offset,
        )




    def set_joint_names(self):
        self._joint_names = self._actuated_wheels + self._actuated_joints

    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n) * 10
        self._limit_acc_j[0, :] = -acc_limit[0 : self.n()]
        self._limit_acc_j[1, :] = acc_limit[0 : self.n()]
