import os
import numpy as np
import pybullet as p
from urdfpy import URDF
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
        joint_info = self.get_indexed_joint_info()
        wheel_joint_names = ["wheel_right_joint", "wheel_left_joint"] 
        torso_joint_name = ["torso_lift_joint"] 
        head_joint_names = ["head_" + str(i) + "_joint" for i in range(3)] 
        arm_right_joint_names = ["arm_right_" + str(i) + "_joint" for i in range(8)] 
        arm_left_joint_names = ["arm_left_" + str(i) + "_joint" for i in range(8)] 
        self._joint_names = ( 
            wheel_joint_names 
            + torso_joint_name 
            + head_joint_names 
            + arm_right_joint_names 
            + arm_left_joint_names 
        ) 
        robot = URDF.load(self._urdf_file) 
        self._urdf_joints = [] 
        for i, joint in enumerate(robot.joints): 
            if joint.name in self._joint_names: 
                self._urdf_joints.append(i) 
        self._robot_joints = [] 
        self._castor_joints = [] 
        for index in joint_info:
            if joint_info[index].decode("UTF-8") in self._joint_names:
                self._robot_joints.append(index)
            if "caster" in joint_info[index].decode("UTF-8"):
                self._castor_joints.append(index)

    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n)
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit
