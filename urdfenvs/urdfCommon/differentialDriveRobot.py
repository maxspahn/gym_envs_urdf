import pybullet as p
import gym
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.genericRobot import GenericRobot


class DifferentialDriveRobot(GenericRobot):
    def __init__(self, n: int, urdfFile: str):
        """Constructorr for differential drive robots.

        Attributes
        ----------

        _wheelRadius : float
            The radius of the actuated wheels.
        _wheelDistance : float
            The distance between the actuated wheels.
        _spawnOffset : np.ndarray
            The offset by which the initial position must be shifted to align
            observation with that position.
        """
        super().__init__(n, urdfFile)
        self._wheelRadius: float = None
        self._wheelDistance: float = None
        self._spawnOffset: np.ndarray = np.array([0.0, 0.0, 0.15])

    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for differential drive robots.
        """
        return self.n() + 1

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> None:
        if hasattr(self, "robot"):
            p.resetSimulation()
        baseOrientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        spawnPos = self._spawnOffset + np.array([pos[0], pos[1], 0.0])
        self.robot = p.loadURDF(
            fileName=self._urdfFile,
            basePosition=spawnPos,
            baseOrientation=baseOrientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        # print(self.getIndexedJointInfo())
        for i in self.castor_joints:
            p.setJointMotorControl2(
                self.robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
            )
        for i in range(2, self._n):
            p.resetJointState(
                self.robot,
                self.robot_joints[i],
                pos[i + 1],
                targetVelocity=vel[i],
            )
        # set base velocity
        self.updateState()
        self._integratedVelocities = vel

    def readLimits(self) -> None:
        robot = URDF.load(self._urdfFile)
        self._limitPos_j = np.zeros((2, self.ns()))
        self._limitVel_j = np.zeros((2, self.ns()))
        self._limitTor_j = np.zeros((2, self.n()))
        self._limitAcc_j = np.zeros((2, self.n()))
        for i in range(self.n()):
            joint = robot.joints[self.urdf_joints[i]]
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
            if i >= 2:
                self._limitPos_j[0, i + 1] = joint.limit.lower
                self._limitPos_j[1, i + 1] = joint.limit.upper
                self._limitVel_j[0, i + 1] = -joint.limit.velocity
                self._limitVel_j[1, i + 1] = joint.limit.velocity
        self._limitVelForward_j = np.array([[-4, -10], [4, 10]])
        self._limitPos_j[0, 0:3] = np.array([-10, -10, -2 * np.pi])
        self._limitPos_j[1, 0:3] = np.array([10, 10, 2 * np.pi])
        self._limitVel_j[0, 0:3] = np.array([-4, -4, -10])
        self._limitVel_j[1, 0:3] = np.array([4, 4, 10])
        self.setAccelerationLimits()

    def getObservationSpace(self) -> gym.spaces.Dict:
        """Gets the observation space for a differential drive robot.

        The observation space is represented as a dictonary. `x` and `xdot`
        denote the configuration position and velocity and `vel` is the current
        forward and rotational velocity of the base. Note that in `xdot`, the
        velocity in Cartesian coordinates is used.
        """
        return gym.spaces.Dict(
            {
                "x": gym.spaces.Box(
                    low=self._limitPos_j[0, :],
                    high=self._limitPos_j[1, :],
                    dtype=np.float64,
                ),
                "vel": gym.spaces.Box(
                    low=self._limitVelForward_j[0, :],
                    high=self._limitVelForward_j[1, :],
                    dtype=np.float64,
                ),
                "xdot": gym.spaces.Box(
                    low=self._limitVel_j[0, :],
                    high=self._limitVel_j[1, :],
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
                self.robot,
                self.robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_acceleration_action(self, accs: np.ndarray, dt: float) -> None:
        """Applies acceleration action to the robot.

        The acceleration action relies on integration of the velocity signal
        and applies velocities at the end. The integrated velocities are
        clipped to avoid very large velocities.
        """
        self._integratedVelocities += dt * accs
        self._integratedVelocities[0] = np.clip(
            self._integratedVelocities[0],
            0.7 * self._limitVelForward_j[0, 0],
            0.7 * self._limitVelForward_j[1, 0],
        )
        self._integratedVelocities[1] = np.clip(
            self._integratedVelocities[1],
            0.5 * self._limitVelForward_j[0, 1],
            0.5 * self._limitVelForward_j[1, 1],
        )
        self.apply_base_velocity(self._integratedVelocities)
        self.apply_velocity_action(self._integratedVelocities)

    def apply_velocity_action_wheels(self, vels: np.ndarray) -> None:
        """Apllies angular velocities to the wheels."""
        for i in range(2):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_base_velocity(self, vels: np.ndarray) -> None:
        """Applies forward and angular velocity to the base.

        The forward and angular velocity of the base is first transformed in
        angular velocities of the wheels using a simple dynamics model.

        """
        velocity_left_wheel = (
            vels[0] + 0.5 * self._wheelDistance * vels[1]
        ) / self._wheelRadius
        velocity_right_wheel = (
            vels[0] - 0.5 * self._wheelDistance * vels[1]
        ) / self._wheelRadius
        wheelVelocities = np.array([velocity_left_wheel, velocity_right_wheel])
        self.apply_velocity_action_wheels(wheelVelocities)

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies angular velocities to the arm joints."""
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def correctBaseOrientation(self, posBase: np.ndarray) -> np.ndarray:
        """Corrects base orientation by -pi.

        The orientation observation should be zero when facing positive
        x-direction. Some robot models are rotated by pi. This is corrected
        here. The function also makes sure that the orientation is always
        between -pi and pi.
        """
        if posBase[2] < -np.pi:
            posBase[2] += 2 * np.pi
        return posBase

    def updateState(self) -> None:
        """Updates the robot state.

        The robot state is stored in the dictonary self.state.  There, the key
        x refers to the translational and rotational position in the world
        frame.  The key xdot referst to the translation and rotational velocity
        in the world frame.  For a differential-drive robot, the forward and
        rotational velocity is stored under the vel key. The value is then a
        numpy array with the values vel_forward and vel_rotational.
        """
        # base position
        linkState = p.getLinkState(self.robot, 0, computeLinkVelocity=0)
        posBase = np.array(
            [
                linkState[0][0],
                linkState[0][1],
                p.getEulerFromQuaternion(linkState[1])[2],
            ]
        )
        # make sure that the rotation is within -pi and pi
        self.correctBaseOrientation(posBase)
        # wheel velocities
        velWheels = p.getJointStates(self.robot, self.robot_joints)
        v_right = velWheels[0][1]
        v_left = velWheels[1][1]
        # simple dynamics model to compute the forward and rotational velocity
        vf = np.array(
            [
                0.5 * (v_right + v_left) * self._wheelRadius,
                (v_right - v_left) * self._wheelRadius / self._wheelDistance,
            ]
        )
        jacobi_nonholonomic = np.array(
            [[np.cos(posBase[2]), 0], [np.sin(posBase[2]), 0], [0, 1]]
        )
        velocityBase = np.dot(jacobi_nonholonomic, vf)
        # joint configurations for holonomic joints
        joint_pos_list = []
        joint_vel_list = []
        for i in range(2, self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        self.state = {
            "x": np.concatenate((posBase, joint_pos)),
            "vel": vf,
            "xdot": np.concatenate((velocityBase, joint_vel)),
        }
