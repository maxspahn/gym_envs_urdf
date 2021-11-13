import pybullet as p
import pybullet_data
import gym
import os
import math
from urdfpy import URDF
import numpy as np


class PointRobot:
    def __init__(self):
        self._n = 3
        self.f_name = os.path.join(os.path.dirname(__file__), 'pointRobot.urdf')
        self.readLimits()

    def reset(self):
        self.robot = p.loadURDF(fileName=self.f_name,
                              basePosition=[0, 0, 0.0])
        # Joint indices as found by p.getJointInfo()
        self.robot_joints = [0, 1, 2]

    def readLimits(self):
        robot = URDF.load(self.f_name)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        for i in range(self._n):
            joint = robot.joints[i]
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j)

    def getTorqueSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
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
        self._friction = 0.0
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction
            )

    def get_ids(self):
        return self.robot

    def apply_torque_action(self, torques):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torques[i])

    def apply_vel_action(self, vels):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def get_observation(self):
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = tuple(joint_pos_list)
        joint_vel = tuple(joint_vel_list)

        # Concatenate position, orientation, velocity
        self.observation = (joint_pos+ joint_vel)
        return self.observation
