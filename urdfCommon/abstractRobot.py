import pybullet as p
import pybullet_data
from abc import ABC, abstractmethod
import gym
from urdfpy import URDF
import numpy as np


class AbstractRobot(ABC):
    def __init__(self, n, fileName):
        self._n = n
        self.fileName = fileName
        self.setJointIndices()
        self.readLimits()
        self._sensors = []

    def n(self):
        return self._n

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

    @abstractmethod
    def setJointIndices(self):
        pass

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

    @abstractmethod
    def setAccLimits(self):
        pass

    def getIndexedJointInfo(self):
        indexedJointInfo = {}
        for i in range(p.getNumJoints(self.robot)):
            jointInfo = p.getJointInfo(self.robot, i)
            indexedJointInfo[jointInfo[0]] = jointInfo[1]
        return indexedJointInfo

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j)

    def getTorqueSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getVelSpaces(self):
        xu = self._limitPos_j[1, :]
        xl = self._limitPos_j[0, :]
        uu = self._limitVel_j[1, :]
        ul = self._limitVel_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getAccSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def disableVelocityControl(self):
        self._friction = 0.0
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction,
            )

    def get_ids(self):
        return self.robot

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
        self.state = np.concatenate((joint_pos, joint_vel))

    def updateSensing(self):
        self.sensor_observation = np.array([])
        for sensor in self._sensors:
            self.sensor_observation = np.append(self.sensor_observation, sensor.sense(self.robot))

    def get_observation(self):
        self.updateState()
        self.updateSensing()
        return np.concatenate((self.state, self.sensor_observation))

    def addSensor(self, sensor):
        self._sensors.append(sensor)
        return sensor.getOSpaceSize()

