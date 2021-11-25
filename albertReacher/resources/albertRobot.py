import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np


class AlbertRobot:
    def __init__(self):
        self._n = 9
        self.f_name = os.path.join(os.path.dirname(__file__), 'albert.urdf')
        self.robot_joints_urdf = [10, 11, 22, 23, 24, 25, 26, 27, 28]
        self.robot_joints_control = [23, 24, 7, 8, 9, 10, 11, 12, 13]
        self.readLimits()

    def n(self):
        return self._n

    def reset(self, pos=None, vel=None):
        self.robot = p.loadURDF(fileName=self.f_name,
                              basePosition=[0, 0, 0.05])
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        for i in [21, 22]:
            p.setJointMotorControl2(
                self.robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0
            )

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j, self._limitAcc_j)

    def readLimits(self):
        robot = URDF.load(self.f_name)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitAcc_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        for i, j in enumerate(self.robot_joints_urdf):
            joint = robot.joints[j]
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
        accLimit = np.array([1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def getTorqueSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def getAccSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def getVelSpaces(self):
        xu = self._limitPos_j[1, :]
        xl = self._limitPos_j[0, :]
        uu = self._limitVel_j[1, :]
        ul = self._limitVel_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def disableVelocityControl(self):
        self._friction = 0
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints_control[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction
            )

    def get_ids(self):
        return self.robot

    def apply_torque_action(self, torques):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints_control[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torques[i])

    def apply_acc_action(self, accs):
        q = []
        qdot = []
        for i in range(2, self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.robot_joints_control[i])
            q.append(pos)
            qdot.append(vel)
        qddot = list(accs)
        q = list(q)
        qdot = list(qdot)
        # TODO: Fails probable due to mismatch in dimension <29-09-21, mspahn> #
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        self.apply_torque_action(tau)

    def apply_vel_action_wheels(self, vels):
        for i in range(2):
            p.setJointMotorControl2(self.robot, self.robot_joints_control[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def apply_vel_action(self, vels):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints_control[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def get_observation(self):
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.robot_joints_control[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # Concatenate position, orientation, velocity
        return np.concatenate((joint_pos, joint_vel))
