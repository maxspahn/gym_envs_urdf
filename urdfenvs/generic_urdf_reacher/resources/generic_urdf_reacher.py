import numpy as np
from urdfenvs.urdfCommon.holonomic_robot import HolonomicRobot


class GenericUrdfReacher(HolonomicRobot):
    def __init__(self, urdf):
        self._urdf = urdf
        super().__init__(-1, self._urdf)

    def set_joint_names(self):
        # TODO Replace it with a automated extraction
        self._joint_names = [joint.name for joint in self._urdf_robot._actuated_joints]

    def set_acceleration_limits(self):
        acc_limit = np.array(
            [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0]
        )
        self._limit_acc_j[0, :] = -acc_limit[0 : self.n()]
        self._limit_acc_j[1, :] = acc_limit[0 : self.n()]
