import os
import numpy as np

from urdfenvs.urdfCommon.differential_drive_robot import DifferentialDriveRobot

class BraitenbergRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 2
        urdf_file = os.path.join(os.path.dirname(__file__), 'braitenberg.urdf')
        super().__init__(n, urdf_file)
        self._wheel_radius = 0.08
        self._wheel_distance = 0.494

    def set_joint_indices(self):
        self._urdf_joints = [2, 3]
        self._robot_joints = [4, 5]
        self._castor_joints = [2, 3]

    def set_acceleration_limits(self):
        acc_limit = np.array([1.0, 1.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

    def correct_base_orientation(self, pos_base):
        pos_base[2] -= np.pi / 2.0
        if pos_base[2] < -np.pi:
            pos_base[2] += 2 * np.pi
        return pos_base

    def apply_base_velocity(self, vels: np.ndarray) -> None:
         """Applies left and right wheel velocity to the base.
 
         This function overwrites the default behavior of the differential drive.
         """
         velocity_left_wheel = vels[0] / self._wheel_radius
         velocity_right_wheel = vels[1] / self._wheel_radius

         wheel_velocities = np.array([velocity_left_wheel, velocity_right_wheel])
         self.apply_velocity_action_wheels(wheel_velocities)
