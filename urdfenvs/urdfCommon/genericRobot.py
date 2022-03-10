import pybullet as p
from abc import ABC, abstractmethod
import gym
import numpy as np


from urdfenvs.sensors.sensor import Sensor


class GenericRobot(ABC):
    """GenericRobot."""

    def __init__(self, n: int, urdfFile: str):
        """Constructor for generic robot.

        Parameters
        ----------

        n: int : Degrees of freedom of the robot
        urdfFile: str : Full path to urdffile
        """
        self._n: int = n
        self._urdfFile: str = urdfFile

        self.setJointIndices()
        self.readLimits()
        self._sensors = []

    def n(self) -> int:
        return self._n

    @abstractmethod
    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Resets the robot to an initial state.

        Parameters
        ----------
        pos: np.ndarray : Initial joint positions
        vel: np.ndarray : Initial joint velocities

        """
        pass

    @abstractmethod
    def setJointIndices(self) -> None:
        """Sets joint indices for urdf parsing.
        
        The urdf file is used to control the robot and
        to read the limits. Control is done using pybullet
        and reading the limits is urdfpy. The index counting
        is different for both so two lists need to be set
        for each robot, self.robot_joints for control
        and self.urdf_joints for reading the limits.
        When castor wheels are present, self.castor_joints
        are also specified.

        """
        pass

    @abstractmethod
    def readLimits(self) -> None:
        """Read and set the joint limits."""
        pass

    @abstractmethod
    def setAccelerationLimits(self):
        pass

    def getIndexedJointInfo(self) -> dict:
        """Get indexed joint info.
        
        This function can be used for debugging and finding
        the correct joint indices for self.setJointIndices.

        """
        indexedJointInfo = {}
        for i in range(p.getNumJoints(self.robot)):
            jointInfo = p.getJointInfo(self.robot, i)
            indexedJointInfo[jointInfo[0]] = jointInfo[1]
        return indexedJointInfo

    def getObservationSpace(self) -> gym.spaces.Dict:
        """Get observation space."""
        return gym.spaces.Dict(
            {
                "x": gym.spaces.Box(
                    low=self._limitPos_j[0, :],
                    high=self._limitPos_j[1, :],
                    dtype=np.float64,
                ),
                "xdot": gym.spaces.Box(
                    low=self._limitVel_j[0, :],
                    high=self._limitVel_j[1, :],
                    dtype=np.float64,
                ),
            }
        )

    def getTorqueSpaces(self) -> tuple:
        """Get observation space and action space when using torque control."""
        ospace = self.getObservationSpace()
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getVelocitySpaces(self) -> tuple:
        """Get observation space and action space when using velocity control."""
        ospace = self.getObservationSpace()
        uu = self._limitVel_j[1, :]
        ul = self._limitVel_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getAccelerationSpaces(self) -> tuple:
        """Get observation space and action space when using acceleration control."""
        ospace = self.getObservationSpace()
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def disableVelocityControl(self):
        """Disables velocity control for all controlled joints.
        
        By default, pybullet uses velocity control. This has to be disabled if
        torques should be directly controlled.
        See func:`~urdfenvs.urdfCommon.genericRobot.genericRobot.apply_torque_action`
        """
        self._friction = 0.0
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction,
            )

    @abstractmethod
    def apply_torque_action(self, torques) -> None:
        pass

    @abstractmethod
    def apply_velocity_action(self, vels) -> None:
        pass

    @abstractmethod
    def apply_acceleration_action(self, accs) -> None:
        pass

    @abstractmethod
    def updateState(self) -> None:
        """Updates the state of the robot.
        
        This function reads current joint position and velocities from the
        pyhsices engine.

        """
        pass

    def updateSensing(self) -> None:
        """Updates the sensing of the robot's sensors."""
        self.sensor_observation = {}
        for sensor in self._sensors:
            self.sensor_observation[sensor.name()] = sensor.sense(self.robot)

    def get_observation(self) -> dict:
        """Updates all observation and concatenate joint states and sensor observation."""
        self.updateState()
        self.updateSensing()
        return {**self.state, **self.sensor_observation}

    def addSensor(self, sensor: Sensor) -> int:
        """Adds sensor to the robot."""
        self._sensors.append(sensor)
        return sensor.getOSpaceSize()

    def sensors(self) -> list:
        return self._sensors
