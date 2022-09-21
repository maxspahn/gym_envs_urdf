import os
import numpy as np

from urdfenvs.urdfCommon.quadrotor import QuadrotorModel


class IRIS(QuadrotorModel):
    """IRIS model

    Attributes
    ---------
    _scaling: float
        The size scaling in which the urdf should be spawned.
        This also effects the dynamics of the system.

    """

    def __init__(self):
        n = 3
        urdf_file = os.path.join(os.path.dirname(__file__), 'iris.urdf')
        self._scaling: float = 1.0
        super().__init__(n, urdf_file)
        self._arm_length    = 0.046
        self._k_thrust      = 5.5716e-06
        self._k_drag        = 1.367e-07
        self._rotor_max_rpm = 2500
        self._rotor_min_rpm = 0
        self._spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15])

    def set_joint_names(self) -> None:
        """Set joint indices.
        """
        self._robot_joints = [1, 2, 3, 4]
    
    def set_acceleration_limits(self):
        acc_limit = np.array([5.0, 5.0, 5.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit
