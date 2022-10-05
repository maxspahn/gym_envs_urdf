import os
import numpy as np

from urdfenvs.urdf_common.bicycle_model import BicycleModel


class Prius(BicycleModel):
    """Prius robot model.

    Attributes
    ---------
    _scaling: float
        The size scaling in which the urdf should be spawned.
        This also effects the dynamics of the system.

    """
    def __init__(self, mode: str):
        n = 2
        urdf_file = os.path.join(os.path.dirname(__file__), 'prius.urdf')
        self._scaling: float = 0.3
        super().__init__(n, urdf_file, mode)
        self._wheel_radius = 0.31265
        self._wheel_distance = 0.494
        self._spawn_offset: np.ndarray = np.array([-0.435, 0.0, 0.05])

    def set_joint_names(self):
        """Set joint indices.

        For the bicycle model robots, the steering joints and the forward joints
        are determined.
        """
        self._robot_joints = [2, 4]
        self._steering_joints = [2, 4]
        self._forward_joints = [3, 5, 6, 7]

    def set_acceleration_limits(self):
        acc_limit = np.array([1.0, 1.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

    def check_state(self, pos, vel):
        if (
            not isinstance(pos, np.ndarray)
            or not pos.size == self.n() + 1
        ):
            pos = np.zeros(self.n() + 1)
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel
