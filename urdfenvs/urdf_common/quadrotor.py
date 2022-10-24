import pybullet as p
import gym
import numpy as np
import logging


from urdfenvs.urdf_common.generic_robot import GenericRobot


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

    def __init__(self, n: int, urdf_file: str, mode: str) -> None:
        """Constructor for quadrotor model robot."""
        super().__init__(n, urdf_file)
        self._swawn_offset: np.ndarray = np.array(
            [0.0, 0.0, 0.15])  # TODO: check this value

        self._pos = np.zeros(3,dtype=np.float32)
        self._quat = np.array([0., 0., 0., 1.], dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._omega = np.zeros(3, dtype=np.float32)
        self._rotor_velocity = np.zeros(4, dtype=np.float32)
        self.state = {"joint_state": {"pose": np.zeros(7, dtype=np.float32), "velocity":
            np.zeros(6, dtype=np.float32), "rotor_velocity": np.zeros(4, dtype=np.float32)}}

    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for quadrotor models.
        
        """
        return 6

    def reset(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            mount_position: np.ndarray,
            mount_orientation: np.ndarray,) -> None:
        """ Reset simulation and add robot """
        logging.warning(
            "The argument 'mount_position' and 'mount_orientation' are \
ignored for drones."
        )
        if hasattr(self, "_robot"):
            p.resetSimulation()
        base_orientation = pos[3:7]
        spawn_pos = self._spawn_offset + np.array([pos[0], pos[1], pos[2]])
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
        for i in range(self.n()):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                0,
                targetVelocity=vel[i],
            )
        # set base velocity
        self.update_state()

    def check_state(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters state of the robot and returns a valid state."""

        if (
            not isinstance(pos, np.ndarray)
            or not pos.size == self.n() + 3
        ):
            pos = np.zeros(self.n() + 3)
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel

    def read_limits(self) -> None:
        self._limit_pos_j = np.zeros((2, self.ns() + 1))
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
        
        # rotor limits
        self._limit_rotors_j = np.zeros((2, 4))
        self._limit_rotors_j[0, :] = np.ones(4) * self._rotor_min_rpm
        self._limit_rotors_j[1, :] = np.ones(4) * self._rotor_max_rpm

    def get_observation_space(self) -> gym.spaces.Dict:
        """Gets the observation space for the quadrotor model.

        The observation space is represented as a dictionary.
        `pose` denotes the pose of the quadrotor in the world frame.
        `velocity` denotes the velocity of the quadrotor in the world frame.
        `rotor_velocities; denotes the rotor velocities.
        """
        return gym.spaces.Dict(
            {
                "joint_state": gym.spaces.Dict({
                    "pose": gym.spaces.Box(
                        low=self._limit_pos_j[0, :],
                        high=self._limit_pos_j[1, :],
                        dtype=np.float32,
                    ),
                    "velocity": gym.spaces.Box(
                        low=self._limit_vel_j[0, :],
                        high=self._limit_vel_j[1, :],
                        dtype=np.float32,
                    ),
                    "rotor_velocity": gym.spaces.Box(
                        low=self._limit_rotors_j[0,:],
                        high=self._limit_rotors_j[1,:],
                        dtype=np.float32,
                    ),
                }),
            }
        )

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies the propeller speed action to the quadrotor.
        """
        direction = np.array([1, 1, -1, -1])
        for i in range(4):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i] * direction[i],
            )
        self._rotor_velocity = vels.astype('float32')

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
        thrusts = self._k_thrust * np.square(rate)

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
            [link_state[0][0], link_state[0][1], link_state[0][2]], dtype=np.float32
        )
        self._quat = np.array(link_state[1], dtype=np.float32)
        self._vel = np.array(link_state[6], dtype=np.float32)
        self._omega = np.array(link_state[7], dtype=np.float32)

        self.state['joint_state']['pose'] = np.concatenate((self._pos, self._quat))
        self.state['joint_state']['velocity'] = np.concatenate((self._vel, self._omega))
        self.state['joint_state']['rotor_velocity'] = self._rotor_velocity

