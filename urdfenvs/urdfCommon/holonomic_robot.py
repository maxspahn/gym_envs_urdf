import pybullet as p
import gym
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.generic_robot import GenericRobot


class HolonomicRobot(GenericRobot):
    """Generic holonomic robot."""
    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        if hasattr(self, "_robot"):
            p.resetSimulation()
        self._robot = p.loadURDF(
            fileName=self._urdf_file,
            basePosition=[0.0, 0.0, 0.0],
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        for i in range(self._n):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                pos[i],
                targetVelocity=vel[i],
            )
        self.update_state()
        self._integrated_velocities = vel

    def read_limits(self) -> None:
        robot = URDF.load(self._urdf_file)
        self._limit_pos_j = np.zeros((2, self._n))
        self._limit_vel_j = np.zeros((2, self._n))
        self._limit_tor_j = np.zeros((2, self._n))
        self._limit_acc_j = np.zeros((2, self._n))
        for i, j in enumerate(self._urdf_joints):
            joint = robot.joints[j]
            self._limit_pos_j[0, i] = joint.limit.lower
            self._limit_pos_j[1, i] = joint.limit.upper
            self._limit_vel_j[0, i] = -joint.limit.velocity
            self._limit_vel_j[1, i] = joint.limit.velocity
            self._limit_tor_j[0, i] = -joint.limit.effort
            self._limit_tor_j[1, i] = joint.limit.effort
        self.set_acceleration_limits()

    def get_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "x": gym.spaces.Box(
                    low=self._limit_pos_j[0, :],
                    high=self._limit_pos_j[1, :],
                    dtype=np.float64,
                ),
                "xdot": gym.spaces.Box(
                    low=self._limit_vel_j[0, :],
                    high=self._limit_vel_j[1, :],
                    dtype=np.float64,
                ),
            }
        )

    def apply_torque_action(self, torques: np.ndarray) -> None:
        for i in range(self._n):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        for i in range(self._n):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_acceleration_action(self, accs: np.ndarray, dt: float) -> None:
        self._integrated_velocities += dt * accs
        self.apply_velocity_action(self._integrated_velocities)

    def update_state(self) -> None:
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(self._n):
            pos, vel, _, _ = p.getJointState(self._robot, self._robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # Concatenate position, orientation, velocity
        self.state = {"x": joint_pos, "xdot": joint_vel}
