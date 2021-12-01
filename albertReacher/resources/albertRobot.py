import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class AlbertRobot(AbstractRobot):
    def __init__(self):
        n = 9
        fileName = os.path.join(os.path.dirname(__file__), 'albert.urdf')
        self._r = 0.08
        self._l = 0.494
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.urdf_joints = [10, 11, 22, 23, 24, 25, 26, 27, 28]
        self.robot_joints = [23, 24, 7, 8, 9, 10, 11, 12, 13]

    def n(self):
        return self._n

    def reset(self, pos=None, vel=None):
        if hasattr(self, "robot"):
            p.resetSimulation()
        baseOrientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        self.robot = p.loadURDF(
            fileName=self.fileName,
            basePosition=[pos[0], pos[1], 0.05],
            baseOrientation=baseOrientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        for i in [21, 22]:
            p.setJointMotorControl2(
                self.robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0
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
        self._vels_int = np.concatenate((self.state[-2:], self.state[13:20]))

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def apply_torque_action(self, torques):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torques[i])

    def apply_acc_action(self, accs, dt):
        self._vels_int += dt * accs
        self.apply_base_velocity(self._vels_int)
        self.apply_vel_action(self._vels_int)

    def apply_vel_action_wheels(self, vels):
        for i in range(2):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def apply_base_velocity(self, vels):
        vel_left = (vels[0] - 0.5 * self._l * vels[1]) / self._r
        vel_right = (vels[0] + 0.5 * self._l * vels[1]) / self._r
        wheelVels = np.array([vel_right, vel_left])
        self.apply_vel_action_wheels(wheelVels)

    def apply_vel_action(self, vels):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

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

        # Concatenate position[0:10], velocity[0:10], vf[0:3]
        self.state = np.concatenate((posBase, joint_pos, velBase, joint_vel, vf))

