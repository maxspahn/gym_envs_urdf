from itertools import chain
import os
import numpy as np
from urdfpy import URDF
from urdfenvs.urdfCommon.holonomic_robot import HolonomicRobot


class DualArmRobot(HolonomicRobot):
    """ Dual arm robot.

    This class implements a toy dual arm robot build from cylinders and
    spheres. It is to test motion planning methods on systems with two
    end-effectors. The kinematic structure can be visualized as
            j1
            |
        ---------
        j2      j4
        |       |
        j3      j5
        |       |

    where j<i> stands for the individual joints.
    """
    def __init__(self):
        n = 5
        urdf_file = os.path.join(
            os.path.dirname(__file__), "dual_arm.urdf"
        )
        super().__init__(n, urdf_file)

    def set_joint_indices(self):
        self._joint_names = ["joint" + str(i) for i in chain(range(1,4),range(5,7))]
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
