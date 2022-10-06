import numpy as np
from urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
import os


class GenericUrdfReacher(HolonomicRobot):
    def __init__(self, urdf, mode):
        self._urdf = urdf

        # search for urdf in package if not found in cwd
        if not os.path.exists(urdf):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            urdf = None
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file == self._urdf:
                        urdf = os.path.join(root, file)
            if urdf is None:
                raise Exception(f"the request urdf {self._urdf} can not be found")
            self._urdf = urdf

        super().__init__(-1, self._urdf, mode=mode)

    def set_joint_names(self):
        # TODO Replace it with a automated extraction
        self._joint_names = [joint.name for joint in self._urdf_robot._actuated_joints]

    def set_acceleration_limits(self):
        acc_limit = np.array(
            [1.0, 1.0, 15.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0]
        )
        self._limit_acc_j[0, :] = -acc_limit[0 : self.n()]
        self._limit_acc_j[1, :] = acc_limit[0 : self.n()]

    def check_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.n():
            center_position = (self._limit_pos_j[0] + self._limit_pos_j[1])/2
            pos = center_position
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel
