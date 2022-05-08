import os
import numpy as np
from urdfpy import URDF
from urdfenvs.urdfCommon.holonomic_robot import HolonomicRobot


class PointRobot(HolonomicRobot):
    def __init__(self):
        urdf_file = os.path.join(os.path.dirname(__file__), "pointRobot.urdf")
        n = 3
        super().__init__(n, urdf_file)

    def set_joint_indices(self):
        self._joint_names = ["mobile_joint_x","mobile_joint_y","mobile_joint_theta"]
        robot = URDF.load(self._urdf_file) 
        self._urdf_joints = [] 
        for i, joint in enumerate(robot.joints): 
            if joint.name in self._joint_names: 
                self._urdf_joints.append(i) 
        self.get_indexed_joint_info()

    def set_acceleration_limits(self):
        acc_limit = np.ones(self._n)
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit
