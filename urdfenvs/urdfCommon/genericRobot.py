import pybullet as p
from abc import ABC, abstractmethod
import gym
import numpy as np


class GenericRobot(ABC):
    def __init__(self, n, fileName):
        self._n = n
        self.fileName = fileName
        self.setJointIndices()
        self.readLimits()
        self._sensors = []

    def n(self):
        return self._n

    @abstractmethod
    def reset(self, pos, vel):
        pass

    @abstractmethod
    def setJointIndices(self):
        pass

    @abstractmethod
    def readLimits(self):
        pass

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
        ospace = self.getObservationSpace()
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getObservationSpace(self):
        return gym.spaces.Dict({
            'x': gym.spaces.Box(low=self._limitPos_j[0, :], high=self._limitPos_j[1, :], dtype=np.float64), 
            'xdot': gym.spaces.Box(low=self._limitVel_j[0, :], high=self._limitVel_j[1, :], dtype=np.float64), 
        })

    def getVelSpaces(self):
        ospace = self.getObservationSpace()
        uu = self._limitVel_j[1, :]
        ul = self._limitVel_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getAccSpaces(self):
        ospace = self.getObservationSpace()
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
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

    @abstractmethod
    def apply_torque_action(self, torques):
        pass

    @abstractmethod
    def apply_vel_action(self, vels):
        pass

    @abstractmethod
    def apply_acc_action(self, accs):
        pass

    @abstractmethod
    def updateState(self):
        pass

    def updateSensing(self):
        self.sensor_observation = {}
        for sensor in self._sensors:
            self.sensor_observation[sensor.name()] = sensor.sense(self.robot)

    def get_observation(self):
        self.updateState()
        self.updateSensing()
        return {**self.state, **self.sensor_observation}

    def addSensor(self, sensor):
        self._sensors.append(sensor)
        return sensor.getOSpaceSize()

    def getSensors(self):
        return self._sensors
