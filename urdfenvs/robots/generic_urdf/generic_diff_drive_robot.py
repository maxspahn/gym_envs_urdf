from typing import List
import numpy as np
from urdfenvs.urdf_common.generic_robot import ControlMode
from urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
from urdfenvs.urdf_common.differential_drive_robot import DifferentialDriveRobot

class GenericDiffDriveRobot(DifferentialDriveRobot):

    def __init__(
            self,
            urdf: str,
            mode: ControlMode,
            actuated_wheels: List[str],
            castor_wheels: List[str],
            wheel_radius: float,
            wheel_distance: float,
            spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15]),
            spawn_rotation: float = 0.0,
            facing_direction: str = 'x',
            not_actuated_joints: List[str] = [],
    ):
        super().__init__(-1,
            urdf,
                         mode,
                         actuated_wheels,
                         castor_wheels,
                         wheel_radius,
                         wheel_distance,
                         spawn_offset=spawn_offset,
                         spawn_rotation=spawn_rotation,
                         facing_direction=facing_direction,
                         not_actuated_joints=not_actuated_joints)

    def set_joint_names(self):
        self._joint_names = []
        for joint in self._urdf_robot._actuated_joints:
            if joint.name in self._castor_wheels:
                continue
            if joint.name in self._actuated_wheels:
                continue
            if joint.name in self._not_actuated_joints:
                continue
            self._joint_names.append(joint.name)
        self._joint_names = self._actuated_wheels + self._joint_names

    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n) * 10
        self._limit_acc_j[0, :] = -acc_limit[0 : self.n()]
        self._limit_acc_j[1, :] = acc_limit[0 : self.n()]

    def check_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.ns():
            center_position = (self._limit_pos_j[0] + self._limit_pos_j[1])/2
            pos = center_position
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel
