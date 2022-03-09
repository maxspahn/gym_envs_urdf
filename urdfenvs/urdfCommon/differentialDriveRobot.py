import pybullet as p
import gym
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.genericRobot import GenericRobot


class DifferentialDriveRobot(GenericRobot):
    def __init__(self, n, urdfFile):
        super().__init__(n, urdfFile)
        self._wheelRadius = None
        self._wheelDistance = None

    def ns(self):
        return self.n() + 1

    def reset(self, pos=None, vel=None):
        if hasattr(self, "robot"):
            p.resetSimulation()
        baseOrientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        self.robot = p.loadURDF(
            fileName=self._urdfFile,
            basePosition=[pos[0], pos[1], 0.15],
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

    def readLimits(self):
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
        self.setAccelerationsLimits()

    def getObservationSpace(self):
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

    def apply_torque_action(self, torques):
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_acceleration_action(self, accs, dt):
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

    def apply_velocity_action_wheels(self, vels):
        for i in range(2):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_base_velocity(self, vels):
        velocity_left_wheel = (vels[0] - 0.5 * self._wheelDistance * vels[1]) / self._wheelRadius
        velocity_right_wheel = (vels[0] + 0.5 * self._wheelDistance * vels[1]) / self._wheelRadius
        wheelVelocities = np.array([velocity_left_wheel, velocity_right_wheel])
        self.apply_velocity_action_wheels(wheelVelocities)

    def apply_velocity_action(self, vels):
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def updateState(self):
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
        posBase[2] -= np.pi / 2.0
        if posBase[2] < -np.pi:
            posBase[2] += 2 * np.pi
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
