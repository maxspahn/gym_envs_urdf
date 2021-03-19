import pybullet as p
import pybullet_data
import gym
import os
import math
from urdfpy import URDF
import numpy as np


class NLinkRobot:
    def __init__(self, n):
        self._n = n
        self.f_name = os.path.join(os.path.dirname(__file__), 'nlink_' + str(n) + '.urdf')

    def reset(self):
        self.robot = p.loadURDF(fileName=self.f_name,
                              basePosition=[0, 0, 0.1])
        # Joint indices as found by p.getJointInfo()
        self.robot_joints = list(range(1, self._n + 1))
        self._friction = 0.0
        self.disableVelocityControl()

    def getSpaces(self):
        robot = URDF.load(self.f_name)
        limitPos_j = np.zeros((2, self._n))
        limitVel_j = np.zeros((2, self._n))
        limitTor_j = np.zeros((2, self._n))
        for i in range(self._n):
            joint = robot.joints[i+1]
            limitPos_j[0, i] = joint.limit.lower
            limitPos_j[1, i] = joint.limit.upper
            limitVel_j[0, i] = -joint.limit.velocity
            limitVel_j[1, i] = joint.limit.velocity
            limitTor_j[0, i] = -joint.limit.effort
            limitTor_j[1, i] = joint.limit.effort
        xu = np.concatenate((limitPos_j[1, :], limitVel_j[1, :]))
        xl = np.concatenate((limitPos_j[0, :], limitVel_j[0, :]))
        uu = limitTor_j[1, :]
        ul = limitTor_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def disableVelocityControl(self):
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction
            )

    def get_ids(self):
        return self.robot

    def apply_action(self, action):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=action[i])

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
