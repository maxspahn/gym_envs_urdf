import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class TiagoRobot(AbstractRobot):
    def __init__(self):
        n = 19
        self._r = 0.1
        self._l = 0.4044
        fileName = os.path.join(os.path.dirname(__file__), "tiago_dual.urdf")
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.robot_joints = [6, 8, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 37, 38, 39, 40, 41, 42, 43]
        self.urdf_joints = [6, 8, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 38, 39, 40, 41, 42, 43, 44]
        self.caster_joints = [9, 10, 11, 12, 13, 14, 15, 16]
        # TODO: This could be used as a starting point to get joint indices from urdf <01-12-21, mspahn> #
        """
        wheel_joint_names = ["wheel_right_joint", "wheel_left_joint"]
        torso_joint_name = ["torso_lift_joint"]
        head_joint_names = ["head_" + str(i) + "_joint" for i in range(3)]
        arm_right_joint_names = ["arm_right_" + str(i) + "_joint" for i in range(8)]
        arm_left_joint_names = ["arm_left_" + str(i) + "_joint" for i in range(8)]
        self._joint_names = (
            wheel_joint_names
            + torso_joint_name
            + head_joint_names
            + arm_right_joint_names
            + arm_left_joint_names
        )
        robot = URDF.load(self.fileName)
        self.urdf_joints = []
        for i, joint in enumerate(robot.joints):
            if joint.name in self._joint_names:
                self.urdf_joints.append(i)

        self.robot_joints = []
        self.caster_joints = []
        for _id in range(p.getNumJoints(self.robot)):
            joint_name = p.getJointInfo(self.robot, _id)[1].decode("UTF-8")
            if joint_name in self._joint_names:
                self.robot_joints.append(_id)
            if "caster" in joint_name:
                self.caster_joints.append(_id)
        __import__('pdb').set_trace()
        self.robot_joints_gripper = []
        """

    def reset(self, pos=np.zeros(20), vel=np.zeros(19)):
        if hasattr(self, "robot"):
            p.resetSimulation()
        baseOrientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        self.robot = p.loadURDF(
            fileName=self.fileName,
            basePosition=[pos[0], pos[1], 0.15],
            baseOrientation=baseOrientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        # set castor wheel friction to zero
        for i in self.caster_joints:
            p.setJointMotorControl2(
                self.robot, jointIndex=i, controlMode=p.VELOCITY_CONTROL, force=0.0
            )
        # set base velocity
        v = np.zeros(2)
        v[0] = vel[0] + vel[1]
        v[1] = vel[0] - vel[0]
        for i in range(2, self._n):
            p.resetJointState(
                self.robot,
                self.robot_joints[i],
                pos[i + 1],
                targetVelocity=vel[i],
            )
        self.updateState()
        self.apply_vel_action_wheels(v)
        self.apply_vel_action(vel)
        self.state[-2:] = v
        self._vels_int = np.concatenate((self.state[40:42], self.state[23:40]))

    def setAccLimits(self):
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def disableVelocityControl(self, friction):
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=friction,
            )

    def apply_torque_action(self, torques):
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_acc_action(self, accs, dt):
        self._vels_int += dt * accs
        self.apply_base_velocity(self._vels_int)
        self.apply_vel_action(self._vels_int)

    def apply_base_velocity(self, vels):
        vel_left = (vels[0] - 0.5 * self._l * vels[1]) / self._r
        vel_right = (vels[0] + 0.5 * self._l * vels[1]) / self._r
        wheelVels = np.array([vel_right, vel_left])
        self.apply_vel_action_wheels(wheelVels)

    def apply_vel_action_wheels(self, vels):
        for i in range(2):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_vel_action(self, vels):
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def updateState(self):
        # Get Base State
        linkState = p.getLinkState(self.robot, 0, computeLinkVelocity=1)
        posBase = np.array(
            [
                linkState[0][0],
                linkState[0][1],
                p.getEulerFromQuaternion(linkState[1])[2],
            ]
        )
        velBase = np.array([linkState[6][0], linkState[6][1], linkState[7][2]])
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(2, self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # forward and rotational velocity
        vf = np.array([np.sqrt(velBase[0] ** 2 + velBase[1] ** 2), velBase[2]])

        # Concatenate position[0:20], velocity[0:20], vf[0:2]
        self.state = np.concatenate((posBase, joint_pos, velBase, joint_vel, vf))
