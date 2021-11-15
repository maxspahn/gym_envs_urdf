import pybullet as p
import pybullet_data
import gym
import os
import math
from urdfpy import URDF
import numpy as np


class PointRobot:
    def __init__(self):
        self._n = 2
        self.f_name = os.path.join(os.path.dirname(__file__), 'pointRobot.urdf')
        self.readLimits()

    def reset(self, pos=np.zeros(2), vel=np.zeros(2)):
        self.robot_joints = [0, 1]
        self.robot = p.loadURDF(fileName=self.f_name,
                              basePosition=[pos[0], pos[1], 0.0])
        self.apply_vel_action(vel)

    def readLimits(self):
        robot = URDF.load(self.f_name)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        self._limitAcc_j = np.zeros((2, self._n))
        for i in range(self._n):
            joint = robot.joints[i]
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
        accLimit = np.array([1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit[0:self._n]
        self._limitAcc_j[1, :] = accLimit[0:self._n]

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j)

    def setWalls(self, limits=[[-2, -2], [2, 2]]):
        colwallId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 10.0, 0.5])
        wall = [p.createMultiBody(0, colwallId, 10, [limits[0][0], 0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))]
        wall = [p.createMultiBody(0, colwallId, 10, [limits[1][0], 0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))]
        wall = [p.createMultiBody(0, colwallId, 10, [0, limits[0][1], 0.0], p.getQuaternionFromEuler([0, 0, np.pi/2]))]
        wall = [p.createMultiBody(0, colwallId, 10, [0, limits[1][1], 0.0], p.getQuaternionFromEuler([0, 0, np.pi/2]))]

    def addObstacle(self, pos, filename):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(
            fileName=filename,
            basePosition=pos
        )

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

    def getAccSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
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

    def apply_acc_action(self, accs):
        q = []
        qdot = []
        qddot = []
        for i in range(self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.robot_joints[i])
            q.append(pos)
            qdot.append(vel)
        qddot = list(accs)
        q = list(q)
        qdot = list(qdot)
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        self.apply_torque_action(tau)

    def get_observation(self):
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
