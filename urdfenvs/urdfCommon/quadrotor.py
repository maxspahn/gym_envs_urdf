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
    _k_thrust : float
        The thrust coefficient, N/(rad/s)**2
    _k_drag : float
        The drag coefficient, Nm/(rad/s)**2
    _rotor_max_rpm : float
        The maximum rpm of the motor, rad/s
    _rotor_min_rpm : float
        The minimum rpm of the motor, rad/s

    _spawn_offset : np.ndarray
        The offset by which the initial position must be shifted to align
        observation with that position.

    self._pos : np.ndarray
        The position of the quadrotor in the world frame
    self._quat : np.ndarray
        The orientation of the quadrotor in the world frame
    self._vel : np.ndarray
        The linear velocity of the quadrotor in the world frame
    self._omega : np.ndarray
        The angular velocity of the quadrotor in the world frame
    """

    def __init__(self, n: int, urdf_file: str) -> None:
        """Constructor for quadrotor model robot."""
        super().__init__(n, urdf_file)
        self._propeller_radius: float = None
        self._arm_length: float = None
        self._k_thrust: float = None
        self._k_drag: float = None
        self._rotor_max_rpm: float = None
        self._rotor_min_rpm: float = None
        self._swawn_offset: np.ndarray = np.array(
            [0.0, 0.0, 0.15])  # TODO: check this value

        self._pos = np.zeros(3)
        self._quat = np.array([0., 0., 0., 1.])
        self._vel = np.zeros(3)
        self._omega = np.zeros(3)

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

        self.set_joint_names()
        self.extract_joint_ids()
        self.read_limits()
        for i in range(self._n):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                pos[i],
                targetVelocity=vel[i],
            )
        # set base velocity
        self.update_state()

    def read_limits(self) -> None:
        self._limit_pos_j = np.zeros((2, self.n() + 4))
        self._limit_vel_j = np.zeros((2, self.ns()))

        # Position limits (x, y, z)
        self._limit_pos_j[0, 0:3] = np.array([-1000., -1000., 0])
        self._limit_pos_j[1, 0:3] = np.array([1000., 1000., 100.])

        # Quaternion limits
        self._limit_pos_j[0, 3:7] = np.array([-1, -1, -1, -1])
        self._limit_pos_j[1, 3:7] = np.array([1, 1, 1, 1])

        # Velocity limits (x, y, z)
        self._limit_vel_j[0, 0:3] = np.array([-40., -40., -40.])
        self._limit_vel_j[1, 0:3] = np.array([40., 40., 40.])
        
        # body rate limits
        self._limit_vel_j[0, 3:6] = np.array([-10., -10., -10.])
        self._limit_vel_j[1, 3:6] = np.array([10., 10., 10.])

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
                    low=self._limit_pos_j[0, 3:7],
                    high=self._limit_pos_j[1, 3:7],
                    dtype=np.float64,
                ),
                "v": gym.spaces.Box(
                    low=self._limit_vel_j[0, :3],
                    high=self._limit_vel_j[1, :3],
                    dtype=np.float64,
                ),
                "w": gym.spaces.Box(
                    low=self._limit_vel_j[0, 3:6],
                    high=self._limit_vel_j[1, 3:6],
                    dtype=np.float64,
                ),

            }
        )

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies the propeller speed action to the quadrotor.
        """
        for i in range(4):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

        self.apply_thrust(vels)
        # self.apply_drag_effect(vels)  # TODO: check literatures for the drag effect

    def apply_acceleration_action(self, accs: np.ndarray) -> None:
        print("Acceleration control not implemented for quadrotor model.")

    def apply_torque_action(self, torques: np.ndarray) -> None:
        """Applies the torques to the quadrotor model.
        """
        print("Torque action is not available for quadrotor model.")

    def correct_base_orientation(self) -> None:
        pass

    def apply_thrust(self, rate: np.ndarray) -> None:
        """PyBullet implementation of a thrust model
        
        Given the rotor rate, calculate the thrust force and moment. 
        The implementation is following Upenn MEAM 620 project 1..

        Parameters
        ----------
        rate : ndarray
            (4)-shaped array of ints containing the rate values of the 4 motors.        
        """
        direction = np.sign(rate) * np.array([1, 1, -1, -1])
        thrusts = self._k_thrust * np.square(rate) * direction

        k = self._k_drag / self._k_thrust
        l = self._arm_length
        torque_mat = np.array([[1,  1,  1,  1],
                               [0,  l,  0, -l],
                               [-l,  0,  l,  0],
                               [k, -k,  k, -k]])
        u = torque_mat @ thrusts
        torque = u[1:]
        for i in range(4):
            p.applyExternalForce(self._robot,
                                 self._robot_joints[i],
                                 posObj=[0, 0, 0],
                                 forceObj=[0, 0, thrusts[i]],
                                 flags=p.LINK_FRAME)
        p.applyExternalTorque(self._robot,
                              0,
                              torqueObj=torque,
                              flags=p.LINK_FRAME)

    def apply_drag_effect(self, rate: np.ndarray) -> None:
        """PyBullet implementation of a drag model

        Base on the system identification in (Foster, 2015)

        Parameters
        ----------
        rate : ndarray
            (4)-shaped array of ints containing the rate values of the 4 motors.

        """
        base_rot = np.array(
            p.getMatrixFromQuaternion(self._quat)).reshape(3, 3)
        drag_factors = -1 * self._k_drag * np.sum(rate)
        drag = np.dot(base_rot, drag_factors * np.array(self._vel))
        p.applyExternalForce(self._robot,
                             0,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME)

    def update_state(self) -> None:
        """Update the robot state.

        The robot joint_state is stored in the dictionary self.state,
        which contains
        `x`: position, np.array([x, y, z])
        `v`: velocity, np.array([vx, vy, vz])
        `q`: attitude, np.array([qx, qy, qz, qw])
        `w`: angular_velocity, np.array([dr, dp, dy])
        """
        # base position
        link_state = p.getLinkState(self._robot, 0, computeLinkVelocity=1)
        self._pos = np.array(
            [link_state[0][0], link_state[0][1], link_state[0][2]]
        )
        self._quat = np.array(link_state[1])
        self._vel = np.array(link_state[6])
        self._omega = np.array(link_state[7])

        self.state = {
            "x": self._pos,
            "v": self._vel,
            "q": self._quat,
            "w": self._omega,
        }
