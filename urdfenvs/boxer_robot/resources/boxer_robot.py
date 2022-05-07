import os
import numpy as np
from urdfpy import URDF
from urdfenvs.urdfCommon.differential_drive_robot import DifferentialDriveRobot


class BoxerRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 2
        urdf_file = os.path.join(os.path.dirname(__file__), 'boxer.urdf')
        super().__init__(n, urdf_file)
        self._wheel_radius = 0.08
        self._wheel_distance = 0.494

    def set_joint_indices(self):
        wheel_joint_names = ["wheel_right_joint", "wheel_left_joint"] 
        self._joint_names = ( 
            wheel_joint_names 
        ) 
        robot = URDF.load(self._urdf_file) 
        self._urdf_joints = [] 
        for i, joint in enumerate(robot.joints): 
            if joint.name in self._joint_names: 
                self._urdf_joints.append(i) 
        self.get_indexed_joint_info()


    def set_acceleration_limits(self):
        acc_limit = np.array([1.0, 1.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

    def correct_base_orientation(self, pos_base):
        pos_base[2] -= np.pi / 2.0
        if pos_base[2] < -np.pi:
            pos_base[2] += 2 * np.pi
        return pos_base
