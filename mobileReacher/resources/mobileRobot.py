import pybullet as p
import pybullet_data
import gym
import os
import math
from urdfpy import URDF
import numpy as np


class MobileRobot:
    def __init__(self, gripper=False):
        self._gripper = gripper
        if gripper:
            self.f_name = os.path.join(os.path.dirname(__file__), 'mobilePandaWithGripper.urdf')
            self._n = 12
            self.robot_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]
            self.control_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]
        else:
            self.f_name = os.path.join(os.path.dirname(__file__), 'mobilePanda.urdf')
            self._n = 10
            self.robot_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
            self.control_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
        self.readLimits()

    def addObstacle(self, pos, filename):
        f_name = os.path.join(os.path.dirname(__file__), "cylinder.urdf")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(
            filename,
            basePosition=pos
        )

    def n(self):
        return self._n

    def reset(self, pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.501, 0.0, 1.8675, 0.0]), vel=None):
        self.robot = p.loadURDF(fileName=self.f_name,
                              basePosition=[0, 0, 0.0])
        # Joint indices as found by p.getJointInfo()
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.control_joints[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=pos[i])
        print("Bringing to initial position..")
        pre_steps = 100
        for i in range(pre_steps):
            p.stepSimulation()
        print("Reached initial position")

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j, self._limitAcc_j)

    def readLimits(self):
        robot = URDF.load(self.f_name)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitAcc_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        for i, j in enumerate(self.control_joints):
            joint = robot.joints[j]
            print("joint : ", joint.name)
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
        accLimit = np.array([1.0, 1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit[0:self._n]
        self._limitAcc_j[1, :] = accLimit[0:self._n]

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

    def apply_acc_action(self, accs):
        q = []
        qdot = []
        qddot = []
        for i in range(self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.control_joints[i])
            q.append(pos)
            qdot.append(vel)
        qddot = list(accs)
        q = list(q)
        qdot = list(qdot)
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        self.apply_torque_action(tau)

    def apply_vel_action(self, vels):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def moveGripper(self, gripperVel):
        # TODO Why can't I use velocity control here..
        for i in range(2):
            p.setJointMotorControl2(self.robot, self.gripper_joints[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=gripperVel)

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

if __name__ == "__main__":
    mr = MobileRobot()
