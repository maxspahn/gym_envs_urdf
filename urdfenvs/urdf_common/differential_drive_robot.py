import pybullet as p
import gym
import numpy as np
import logging

from urdfenvs.urdf_common.generic_robot import GenericRobot


class DifferentialDriveRobot(GenericRobot):
    """Differential drive robot.

    Attributes
    ----------

    _wheel_radius : float
        The radius of the actuated wheels.
    _wheel_distance : float
        The distance between the actuated wheels.
    _spawn_offset : np.ndarray
        The offset by which the initial position must be shifted to align
        observation with that position.
    """

    def __init__(self, n: int, urdf_file: str, mode: str, number_actuated_axes: int=1):
        """Constructor for differential drive robots."""
        super().__init__(n, urdf_file, mode)
        self._wheel_radius: float = None
        self._wheel_distance: float = None
        self._number_actuated_axes: int = number_actuated_axes
        self._spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15])


    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for differential drive robots.
        """
        return self.n() + 1

    def get_velocity_spaces(self) -> tuple:
        """Get observation space and action space when using velocity
        control.

        Overrides velocity spaces from default, because a differential drive has limits in x,y and
        theta direction, while the action space should be limited to the forward and angular velocity."""
        ospace = self.get_observation_space()
        uu = np.concatenate(
            (self._limit_vel_forward_j[1, :], self._limit_vel_j[1, 3:]), axis=0
        )
        ul = np.concatenate(
            (self._limit_vel_forward_j[0, :], self._limit_vel_j[0, 3:]), axis=0
        )
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def reset(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            mount_position: np.ndarray,
            mount_orientation: np.ndarray,) -> None:
        """ Reset simulation and add robot """
        logging.warning(
            "The argument 'mount_position' and 'mount_orientation' are \
ignored for differential drive robots."
        )
        if hasattr(self, "_robot"):
            p.resetSimulation()
        base_orientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        spawn_position = self._spawn_offset
        spawn_position[0:2] += pos[0:2]
        self._robot = p.loadURDF(
            fileName=self._urdf_file,
            basePosition=spawn_position,
            baseOrientation=base_orientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        self.set_joint_names()
        self.extract_joint_ids()
        self.read_limits()
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        for i in self._castor_joints:
            p.setJointMotorControl2(
                self._robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
            )
        for i in self._castor_joints:
            p.changeDynamics(self._robot, i, lateralFriction=0)
        for i in range(2, self._n):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                pos[i + 1],
                targetVelocity=vel[i],
            )
        # set base velocity
        self.update_state()
        self._integrated_velocities = vel

    def check_state(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters state of the robot and returns a valid state."""

        if not isinstance(pos, np.ndarray) or not pos.size == self.ns():
            pos = np.zeros(self.ns())
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel

    def read_limits(self) -> None:
        """ Set position, velocity, acceleration
        and motor torque lower en upper limits """
        self._limit_pos_j = np.zeros((2, self.ns()))
        self._limit_vel_j = np.zeros((2, self.ns()))
        self._limit_tor_j = np.zeros((2, self.n()))
        self._limit_acc_j = np.zeros((2, self.n()))
        for i in range(self.n()):
            joint = self._urdf_robot.robot.joints[self._urdf_joints[i]]
            self._limit_tor_j[0, i] = -joint.limit.effort
            self._limit_tor_j[1, i] = joint.limit.effort
            if i >= 2:
                self._limit_pos_j[0, i + 1] = joint.limit.lower
                self._limit_pos_j[1, i + 1] = joint.limit.upper
                self._limit_vel_j[0, i + 1] = -joint.limit.velocity
                self._limit_vel_j[1, i + 1] = joint.limit.velocity
        self._limit_vel_forward_j = np.array([[-4.0, -10.0], [4.0, 10.0]])
        self._limit_pos_j[0, 0:3] = np.array([-10.0, -10.0, -2 * np.pi])
        self._limit_pos_j[1, 0:3] = np.array([10.0, 10.0, 2 * np.pi])
        self._limit_vel_j[0, 0:3] = np.array([-4.0, -4.0, -10.0])
        self._limit_vel_j[1, 0:3] = np.array([4.0, 4.0, 10.0])
        self.set_acceleration_limits()

    def get_observation_space(self) -> gym.spaces.Dict:
        """Gets the observation space for a differential drive robot.

        The observation space is represented as a dictionary.
        `join_state` containing:
        `position` the concatenated positions of joints
        in their local configuration space.
        `velocity` the concatenated velocities of joints
        in their local configuration space.
        `forward_velocity` the forward velocity of the robot.
        """
        return gym.spaces.Dict(
            {
                "joint_state": gym.spaces.Dict(
                    {
                        "position": gym.spaces.Box(
                            low=self._limit_pos_j[0, :],
                            high=self._limit_pos_j[1, :],
                            dtype=np.float64,
                        ),
                        "velocity": gym.spaces.Box(
                            low=self._limit_vel_j[0, :],
                            high=self._limit_vel_j[1, :],
                            dtype=np.float64,
                        ),
                        "forward_velocity": gym.spaces.Box(
                            low=np.array([self._limit_vel_forward_j[0][0]]),
                            high=np.array([self._limit_vel_forward_j[1][0]]),
                            shape=(1,),
                            dtype=np.float64,
                        ),
                    }
                )
            }
        )

    def apply_torque_action(self, torques: np.ndarray) -> None:
        """Applies torque action to the arm joints of the robot.

        Torque control is not available for the base at the moment.
        """
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_acceleration_action(self, accs: np.ndarray, dt: float) -> None:
        """Applies acceleration action to the robot.

        The acceleration action relies on integration of the velocity signal
        and applies velocities at the end. The integrated velocities are
        clipped to avoid very large velocities.
        """
        self._integrated_velocities += dt * accs
        self._integrated_velocities[0] = np.clip(
            self._integrated_velocities[0],
            0.7 * self._limit_vel_forward_j[0, 0],
            0.7 * self._limit_vel_forward_j[1, 0],
        )
        self._integrated_velocities[1] = np.clip(
            self._integrated_velocities[1],
            0.5 * self._limit_vel_forward_j[0, 1],
            0.5 * self._limit_vel_forward_j[1, 1],
        )
        self.apply_base_velocity(self._integrated_velocities)
        self.apply_velocity_action(self._integrated_velocities)

    def apply_velocity_action_wheels(self, vels: np.ndarray) -> None:
        """Applies angular velocities to the wheels."""
        for axis in range(self._number_actuated_axes):
            for i in range(2):
                p.setJointMotorControl2(
                    self._robot,
                    self._robot_joints[axis * 2 + i],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=vels[i],
                )

    def apply_base_velocity(self, vels: np.ndarray) -> None:
        """Applies forward and angular velocity to the base.

        The forward and angular velocity of the base
        is first transformed in angular velocities of
        the wheels using a simple dynamics model.

        """
        velocity_left_wheel = (
            vels[0] + 0.5 * self._wheel_distance * vels[1]
        ) / self._wheel_radius
        velocity_right_wheel = (
            vels[0] - 0.5 * self._wheel_distance * vels[1]
        ) / self._wheel_radius

        wheel_velocities = np.array([velocity_left_wheel, velocity_right_wheel])
        self.apply_velocity_action_wheels(wheel_velocities)

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies angular velocities to the arm joints."""
        self.apply_base_velocity(vels) 
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def correct_base_orientation(self, pos_base: np.ndarray) -> np.ndarray:
        """Corrects base orientation by -pi.

        The orientation observation should be zero when facing positive
        x-direction. Some robot models are rotated by pi. This is corrected
        here. The function also makes sure that the orientation is always
        between -pi and pi.
        """
        if pos_base[2] < -np.pi:
            pos_base[2] += 2 * np.pi
        return pos_base

    def update_state(self) -> None:
        """Updates the robot state.

        The robot state is stored in the self.state, which contains
        a dictionary with key 'joint_state' with nested dictionaries:
        `position`: np.array((base_pose2D, joint_position_2, ...,
            joint_position_n))
            the position in local configuration space
            the base's configuration space aligns with the world frame
            base_pose2D = (x_positions, y_position, orientation)
            the joints 2 to n have al 1-dimensional configuration space
            joint_position_i = (position in local configuration space)
        `velocity`: np.array((base_twist2D, joint_velocity_2, ...,
            joint_velocity_n))
            the velocity in local configuration space
            the base's configuration space aligns with the world frame
            base_pose2D = (x_positions, y_position, orientation)
            the joints 2 to n have al one dimensional configuration space
            joint_velocity_i = (position in local configuration space)
        `forward_velocity`: float
            forward velocity in robot frame
        """
        # base position
        link_state = p.getLinkState(self._robot, 0, computeLinkVelocity=0)
        pos_base = np.array(
            [
                link_state[0][0],
                link_state[0][1],
                p.getEulerFromQuaternion(link_state[1])[2],
            ]
        )
        # make sure that the rotation is within -pi and pi
        self.correct_base_orientation(pos_base)
        # wheel velocities
        vel_wheels = p.getJointStates(self._robot, self._robot_joints)
        v_right = vel_wheels[0][1]
        v_left = vel_wheels[1][1]
        # simple dynamics model to compute the forward and angular velocity
        forward_velocity = 0.5 * (v_right + v_left) * self._wheel_radius
        angular_velocity = (
            (v_right - v_left) * self._wheel_radius / self._wheel_distance
        )

        jacobi_nonholonomic = np.array(
            [[np.cos(pos_base[2]), 0], [np.sin(pos_base[2]), 0], [0, 1]]
        )
        velocity_base = np.dot(
            jacobi_nonholonomic, np.array([forward_velocity, angular_velocity])
        )

        # joint configurations for holonomic joints
        joint_pos_list = []
        joint_vel_list = []
        for i in range(2, self._n):
            pos, vel, _, _ = p.getJointState(self._robot, self._robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        self.state = {
            "joint_state": {
                "position": np.concatenate((pos_base, joint_pos)),
                "velocity": np.concatenate((velocity_base, joint_vel)),
                "forward_velocity": np.array([forward_velocity]),
            }
        }
