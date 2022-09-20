import pybullet as p
import gym
import numpy as np


from urdfenvs.urdfCommon.generic_robot import GenericRobot


class QuadrotorModel(GenericRobot):
    """ Quadrotor Model for drones

    Attributes
    ----------

    _propeller_radius : float
        The radius of the actuated propellers
    _arm_length : float
        The length of the arm connecting the propellers to the center of mass
    _gravity : float
        The gravitational constant
    _mass : float
        The mass of the quadrotor
    _inertia : np.ndarray
        The inertia tensor of the quadrotor, [Ixx, Iyy, Izz]
    _Kf : float
        The thrust coefficient, N/(rad/s)**2
    _Kg : float
        The drag coefficient, Nm/(rad/s)**2
    _rotor_max_rpm : float
        The maximum rpm of the motor, rad/s
    _rotor_min_rpm : float
        The minimum rpm of the motor, rad/s

    _spawn_offset : np.ndarray
        The offset by which the initial position must be shifted to align
        observation with that position.

    """

    def __init__(self, n: int, urdf_file: str) -> None:
        """Constructor for quadrotor model robot."""
        super().__init__(n, urdf_file)
        self._propeller_radius: float = None
        self._arm_length: float = None
        self._gravity: float = 9.81
        self._inertia: np.ndarray = np.zeros(3)
        self._mass: float = None
        self._k_thrust: float = None
        self._k_drag: float = None
        self._rotor_max_rpm: float = None
        self._rotor_min_rpm: float = None
        self._swawn_offset: np.ndarray = np.array(
            [0.0, 0.0, 0.15])  # TODO: check this value

    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for quadrotor models.
        """
        return self.n() + 3

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> None:
        if hasattr(self, "_robot"):
            p.resetSimulation()
        base_orientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        spawn_pos = self._spawn_offset + np.array([pos[0], pos[1], 0.0])
        self._robot = p.loadURDF(
            fileName=self._urdf_file,
            basePosition=spawn_pos,
            baseOrientation=base_orientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            globalScaling=self._scaling,
        )

        # precompute inertia
        k = self._k_drag / self._k_thrust
        L = self._arm_length
        self._TM = np.array([[1,  1,  1,  1],
                            [0,  L,  0, -L],
                            [-L,  0,  L,  0],
                            [k, -k,  k, -k]])
        self._inv_inertia = np.linalg.inv(self._inertia)
        self._weight = np.array([0, 0, self._mass * self._gravity])

        self.set_joint_names()
        self.extract_joint_ids()
        self.read_limits()
        # set base velocity
        self.update_state()

    def read_limits(self) -> None:
        self._limit_pos_j = np.zeros((2, self.ns()))

        # Position limits (x, y, z)
        self._limit_pos_j[0, 0:3] = np.array([-1000., -1000., 0])
        self._limit_pos_j[1, 0:3] = np.array([1000., 1000., 100.])

        # Attitude limits (roll, pitch, yaw)
        self._limit_pos_j[0, 3:6] = np.array([-np.pi, -np.pi, -np.pi])
        self._limit_pos_j[1, 3:6] = np.array([np.pi, np.pi, np.pi])

        # TODO

    def get_observation_space(self) -> gym.spaces.Dict:
        """Gets the observation space for the quadrotor model.

        The observation space is represented as a dictionary.
        `x` denotes the position of the quadrotor in the world frame.
        `v` denotes the velocity of the quadrotor in the world frame.
        `q` denotes the orientation (row, pitch, yaw),
        `w` denotes the angular velocity (dr, dp, dy).
        """
        return gym.spaces.Dict(
            {
                "x": gym.spaces.Box(
                    low=self._limit_pos_j[0, :3],
                    high=self._limit_pos_j[1, :3],
                    dtype=np.float64,
                ),
                "q": gym.spaces.Box(
                    low=self._limit_pos_j[0, 3:6],
                    high=self._limit_pos_j[1, 3:6],
                    dtype=np.float64,
                ),
                "v": gym.spaces.Box(
                    low=self._limit_vel_j[0, :],
                    high=self._limit_vel_j[1, :],
                    dtype=np.float64,
                ),
                "w": gym.spaces.Box(
                    low=self._limit_vel_j[0, :],
                    high=self._limit_vel_j[1, :],
                    dtype=np.float64,
                ),

            }
        )

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies the velocity action to the quadrotor.
        """
        pass

    def apply_acceleration_action(self, accs: np.ndarray) -> None:
        pass

    def apply_torque_action(self, torques: np.ndarray) -> None:
        """Applies the torques to the quadrotor model.

        Parameters
        ----------
        torques : np.ndarray
            The torques to be applied to the quadrotor model.
        """
        pass

    def correct_base_orientation(self) -> None:
        pass

    def update_state(self) -> None:
        """Update the robot state.

        The robot joint_state is stored in the dictionary self.state,
        which contains
        `x`: position, np.array([x, y, z])
        `v`: velocity, np.array([vx, vy, vz])
        `q`: attitude, np.array([row, pitch, yaw])
        `w`: angular_velocity, np.array([dr, dp, dy])
        """
        # base position
        link_state = p.getLinkState(self._robot, 0, computeLinkVelocity=1)
        pos_base = np.array(
            [link_state[0][0], link_state[0][1], link_state[0][2]]
        )

        rotor_speed = np.clip(
            thrust_cmd, self._rotor_min_rpm, self._rotor_max_rpm)
        rotor_thrust = self._k_thrust * np.square(rotor_speed)
        self.state = {
            "x": np.array(),
            "v": np.array(),
            "q": np.array(),
            "w": np.array(),
        }

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[0, -s[2], s[1]],
                         [s[2], 0, -s[0]],
                         [-s[1], s[0], 0]])

    @classmethod
    def quat_dot(quat, omega):
        """
        Parameters:
            quaternion, [i,j,k,w]
            omega, angular velocity of body in body axes

        Returns
            quat_dot, [i,j,k,w]

        """
        # Adapted from "Quaternions And Dynamics" by Basile Graf.
        (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
        G = np.array([[q3,  q2, -q1, -q0],
                      [-q2,  q3,  q0, -q1],
                      [q1, -q0,  q3, -q2]])
        quat_dot = 0.5 * G.T @ omega
        # Augment to maintain unit quaternion.
        quat_err = np.sum(quat**2) - 1
        quat_err_grad = 2 * quat
        quat_dot = quat_dot - quat_err * quat_err_grad
        return quat_dot

    @classmethod
    def quat2rpy(q: np.ndarray) -> np.ndarray:
        """Converts quaternion to roll, pitch, yaw
        """
        pass
