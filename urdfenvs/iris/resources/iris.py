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
        self._gravity       = 9.81
        self._arm_length    = 0.046
        self._mass          = 0.030
        self._inertia       = np.array([1.43e-5, 1.43e-5, 2.89e-5])
        self._k_thrust      = 2.3e-08
        self._k_drag        = 7.8e-11
        self._rotor_max_rpm = 2500
        self._rotor_min_rpm = 0

    def set_joint_names(self) -> None:
        """Set joint indices.
        """
        pass
    
    def set_acceleration_limits(self):
        acc_limit = np.array([5.0, 5.0, 5.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit
