import pybullet as p
import gym
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.generic_robot import GenericRobot


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
    def __init__(self, n: int, urdf_file: str):
        """Constructorr for differential drive robots."""
        super().__init__(n, urdf_file)
        self._wheel_radius: float = None
        self._wheel_distance: float = None
        self._spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15])

    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for differential drive robots.
        """
        return self.n() + 1

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
        )
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        # print(self.getIndexedJointInfo())
        for i in self._castor_joints:
            p.setJointMotorControl2(
                self._robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
            )
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

    def read_limits(self) -> None:
        robot = URDF.load(self._urdf_file)
        self._limit_pos_j = np.zeros((2, self.ns()))
        self._limit_vel_j = np.zeros((2, self.ns()))
        self._limit_tor_j = np.zeros((2, self.n()))
        self._limit_acc_j = np.zeros((2, self.n()))
        for i in range(self.n()):
            joint = robot.joints[self._urdf_joints[i]]
            self._limit_tor_j[0, i] = -joint.limit.effort
            self._limit_tor_j[1, i] = joint.limit.effort
            if i >= 2:
                self._limit_pos_j[0, i + 1] = joint.limit.lower
                self._limit_pos_j[1, i + 1] = joint.limit.upper
                self._limit_vel_j[0, i + 1] = -joint.limit.velocity
                self._limit_vel_j[1, i + 1] = joint.limit.velocity
        self._limit_vel_forward_j = np.array([[-4, -10], [4, 10]])
        self._limit_pos_j[0, 0:3] = np.array([-10, -10, -2 * np.pi])
        self._limit_pos_j[1, 0:3] = np.array([10, 10, 2 * np.pi])
        self._limit_vel_j[0, 0:3] = np.array([-4, -4, -10])
        self._limit_vel_j[1, 0:3] = np.array([4, 4, 10])
        self.set_acceleration_limits()

    def get_observation_space(self) -> gym.spaces.Dict:
        """Gets the observation space for a differential drive robot.

        The observation space is represented as a dictonary. `x` and `xdot`
        denote the configuration position and velocity and `vel` is the current
        forward and rotational velocity of the base. Note that in `xdot`, the
        velocity in Cartesian coordinates is used.
        """
        return gym.spaces.Dict(
            {
                "x": gym.spaces.Box(
                    low=self._limit_pos_j[0, :],
                    high=self._limit_pos_j[1, :],
                    dtype=np.float64,
                ),
                "vel": gym.spaces.Box(
                    low=self._limit_vel_forward_j[0, :],
                    high=self._limit_vel_forward_j[1, :],
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
        """Apllies angular velocities to the wheels."""
        for i in range(2):
            p.setJointMotorControl2(
                self._robot,
                self._robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_base_velocity(self, vels: np.ndarray) -> None:
        """Applies forward and angular velocity to the base.

        The forward and angular velocity of the base is first transformed in
        angular velocities of the wheels using a simple dynamics model.

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

        The robot state is stored in the dictonary self.state.  There, the key
        x refers to the translational and rotational position in the world
        frame.  The key xdot referst to the translation and rotational velocity
        in the world frame.  For a differential-drive robot, the forward and
        rotational velocity is stored under the vel key. The value is then a
        numpy array with the values vel_forward and vel_rotational.
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
        # simple dynamics model to compute the forward and rotational velocity
        vf = np.array(
            [
                0.5 * (v_right + v_left) * self._wheel_radius,
                (v_right - v_left) * self._wheel_radius / self._wheel_distance,
            ]
        )
        jacobi_nonholonomic = np.array(
            [[np.cos(pos_base[2]), 0], [np.sin(pos_base[2]), 0], [0, 1]]
        )
        velocity_base = np.dot(jacobi_nonholonomic, vf)
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
            "x": np.concatenate((pos_base, joint_pos)),
            "vel": vf,
            "xdot": np.concatenate((velocity_base, joint_vel)),
        }
