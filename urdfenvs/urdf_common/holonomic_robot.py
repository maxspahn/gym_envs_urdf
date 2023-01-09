import gym
import numpy as np

from urdfenvs.urdf_common.generic_robot import GenericRobot


class HolonomicRobot(GenericRobot):
    """Generic holonomic robot."""

    def reset(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            mount_position: np.ndarray,
            mount_orientation: np.ndarray,) -> None:

        if hasattr(self, "_robot"):
            self._physics_engine.reset_simulation()
        self._robot = self._physics_engine.load_urdf(self._urdf_file, mount_position, mount_orientation)
        self.set_joint_names()
        self.extract_joint_ids()
        self._physics_engine.set_initial_joint_states(self._robot, self._robot_joints, pos, vel)
        self.update_state()
        self._integrated_velocities = vel

    def read_limits(self) -> None:
        """
        Set position, velocity, acceleration and
        motor torque lower en upper limits
        """
        self._limit_pos_j = np.zeros((2, self._n))
        self._limit_vel_j = np.zeros((2, self._n))
        self._limit_tor_j = np.zeros((2, self._n))
        self._limit_acc_j = np.zeros((2, self._n))
        for i, j in enumerate(self._urdf_joints):
            joint = self._urdf_robot.robot.joints[j]
            self._limit_pos_j[0, i] = joint.limit.lower
            self._limit_pos_j[1, i] = joint.limit.upper
            self._limit_vel_j[0, i] = -joint.limit.velocity
            self._limit_vel_j[1, i] = joint.limit.velocity
            self._limit_tor_j[0, i] = -joint.limit.effort
            self._limit_tor_j[1, i] = joint.limit.effort
        self.set_acceleration_limits()

    def get_observation_space(self) -> gym.spaces.Dict:
        """
        Gets the observation space for a holonomic robot.

        The observation space is represented as a dictionary.
        `joint_state` containing:
        `position` the concatenated positions of joints in
        their local configuration space.
        `velocity` the concatenated velocities of joints in
        their local configuration space.
        """
        return gym.spaces.Dict(
            {
                "joint_state": gym.spaces.Dict({
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

                }),
            }
        )

    def apply_torque_action(self, torques: np.ndarray) -> None:
        self._physics_engine.apply_torque_action(torques, self._robot, self._robot_joints)

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        self._physics_engine.apply_velocity_action(vels, self._robot, self._robot_joints)

    def apply_acceleration_action(self, accs: np.ndarray, dt: float) -> None:
        self._integrated_velocities += dt * accs
        self.apply_velocity_action(self._integrated_velocities)

    def update_state(self) -> None:
        """
        Updates the robot joint_state.

        The robot joint_state is stored in the dictionary self.state,
        which contains:
       `position`: np.array([joint_position_0, ..., joint_position_n-1)
           the joints 0 to n-1 have al 1-dimensional configuration space
           joint_position_i = (position in local configuration space)
       `velocity`: np.array([joint_velocity_0, ..., joint_velocity_n-1])
           the joints 0 to n-1 have al one dimensional configuration space
           joint_velocity_i = (position in local configuration space)
       """

        # Get Joint Configurations

        joint_pos, joint_vel = self._physics_engine.joint_states(self._robot, self._robot_joints)

        # Concatenate position, orientation, velocity
        self.state = {"joint_state": {"position": joint_pos,
                      "velocity": joint_vel}}
