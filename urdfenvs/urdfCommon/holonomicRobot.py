import pybullet as p
import pybullet_data
from abc import ABC, abstractmethod
import gym
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.genericRobot import GenericRobot


class HolonomicRobot(GenericRobot):
    def __init__(self, n, fileName):
        super().__init__(n, fileName)

    def reset(self, pos, vel):
        if hasattr(self, "robot"):
            p.resetSimulation()
        self.robot = p.loadURDF(
            fileName=self.fileName,
            basePosition=[0.0, 0.0, 0.0],
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        for i in range(self._n):
            p.resetJointState(
                self.robot,
                self.robot_joints[i],
                pos[i],
                targetVelocity=vel[i],
            )
        self.updateState()


    def readLimits(self):
        robot = URDF.load(self.fileName)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        self._limitAcc_j = np.zeros((2, self._n))
        for i, j in enumerate(self.urdf_joints):
            joint = robot.joints[j]
            print(joint.name)
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
        self.setAccLimits()

    def getObservationSpace(self):
        return gym.spaces.Dict({
            'x': gym.spaces.Box(low=self._limitPos_j[0, :], high=self._limitPos_j[1, :], dtype=np.float64), 
            'xdot': gym.spaces.Box(low=self._limitVel_j[0, :], high=self._limitVel_j[1, :], dtype=np.float64), 
        })

    def apply_torque_action(self, torques):
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_vel_action(self, vels):
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_acc_action(self, accs):
        accs = np.clip(accs, self._limitAcc_j[0, :], self._limitAcc_j[1, :])
        q = []
        qdot = []
        for i in range(self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.robot_joints[i])
            q.append(pos)
            qdot.append(vel)
        qddot = list(accs)
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        self.apply_torque_action(tau)

    def updateState(self):
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # Concatenate position, orientation, velocity
        self.state = {'x': joint_pos, 'xdot': joint_vel}
