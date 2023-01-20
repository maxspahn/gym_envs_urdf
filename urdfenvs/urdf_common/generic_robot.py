import pybullet as p
from abc import ABC, abstractmethod
import gym
import numpy as np
import yourdfpy
from urdfenvs.sensors.sensor import Sensor
from enum import Enum
from typing import List

class ControlMode(Enum):
    torque = 'tor'
    acceleration = 'acc'
    velocity = 'vel'

class GenericRobot(ABC):
    """GenericRobot."""

    def __init__(self, n: int, urdf_file: str, mode=ControlMode.velocity):
        """Constructor for generic robot.

        Parameters
        ----------

        n: int : Degrees of freedom of the robot
        urdf_file: str : Full path to urdf file
        """
        self._urdf_file: str = urdf_file
        self._sensors: List[Sensor] = []
        self._urdf_robot = yourdfpy.urdf.URDF.load(self._urdf_file)
        self._mode = ControlMode(mode)
        if n > 0:
            self._n = n
        else:
            self._n: int = self._urdf_robot.num_actuated_joints
        self.set_joint_names()
        self.extract_joint_ids()
        self.read_limits()

    def n(self) -> int:
        return self._n

    def ns(self) -> int:
        return self.n()

    @abstractmethod
    def reset(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            mount_position: np.ndarray,
            mount_orientation: np.ndarray,) -> None:
        """Resets the robot to an initial state.

        Parameters
        ----------
        pos: np.ndarray:
            Initial joint positions
        vel: np.ndarray:
            Initial joint velocities
        mount_position: np.ndarray:
            Mounting position
        mount_orientation: np.ndarray:
            Mounting orientation

        """
        pass

    @abstractmethod
    def set_joint_names(self) -> None:
        """Sets joint indices for urdf parsing.

        Input the names of joints manually.

        """
        pass

    @abstractmethod
    def read_limits(self) -> None:
        """Read and set the joint limits."""
        pass

    @abstractmethod
    def set_acceleration_limits(self):
        pass

    def extract_joint_ids(self) -> None:
        """Automated extraction of joint ids

        Extract joint ids by the joint names.

        """
        if not hasattr(self, "_joint_names"):
            return
        self._urdf_joints = []
        for i, joint_name in enumerate(self._urdf_robot.joint_names):
            if joint_name in self._joint_names:
                self._urdf_joints.append(i)
        if hasattr(self, "_robot"):
            self._robot_joints = []
            self._castor_joints = []
            num_joints = p.getNumJoints(self._robot)
            for name in self._joint_names:
                for i in range(num_joints):
                    joint_info = p.getJointInfo(self._robot, i)
                    joint_name = joint_info[1].decode("UTF-8")
                    if joint_name == name:
                        self._robot_joints.append(i)
                for i in range(num_joints):
                    joint_info = p.getJointInfo(self._robot, i)
                    joint_name = joint_info[1].decode("UTF-8")
                    if "castor" in joint_name:
                        self._castor_joints.append(i)


    def get_observation_space(self) -> gym.spaces.Dict:
        """Get observation space."""
        return gym.spaces.Dict(
            {
                "joint_state": gym.spaces.Dict(
                    {
                        "position": gym.spaces.Box(
                        low=self._limit_pos_j[0, :],
                        high=self._limit_pos_j[1, :],
                        dtype=np.float64,
                    ),
                        "velocity": gym.spaces.Box(
                            low=self._limit_vel_j[0, :],
                            high=self._limit_vel_j[1, :],
                            dtype=np.float64,
                        ),
                    }
                )
            }
        )

    def check_state(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters state of the robot and returns a valid state."""

        if not isinstance(pos, np.ndarray) or not pos.size == self.n():
            pos = np.zeros(self.n())
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel


    def get_torque_spaces(self) -> tuple:
        """Get observation space and action space when using torque control."""
        ospace = self.get_observation_space()
        uu = self._limit_tor_j[1, :]
        ul = self._limit_tor_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def get_velocity_spaces(self) -> tuple:
        """Get observation space and action space when using velocity
        control."""
        ospace = self.get_observation_space()
        uu = self._limit_vel_j[1, :]
        ul = self._limit_vel_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def get_acceleration_spaces(self) -> tuple:
        """Get observation space and action space when using acceleration
        control."""
        ospace = self.get_observation_space()
        uu = self._limit_acc_j[1, :]
        ul = self._limit_acc_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def disable_velocity_control(self):
        """Disables velocity control for all controlled joints.

        By default, pybullet uses velocity control. This has to be disabled if
        torques should be directly controlled.  See
        func:`~urdfenvs.urdfCommon.generic_robot.generic_rob
        ot.apply_torque_action`
        """
        self._friction = 0.0
        for i in range(self._n):
            p.setJointMotorControl2(
                self._robot,
                jointIndex=self._robot_joints[i],
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

    def apply_action(self, action, dt=None) -> None:
        if self._mode == ControlMode.torque:
            self.apply_torque_action(action)
        elif self._mode == ControlMode.velocity:
            self.apply_velocity_action(action)
        elif self._mode == ControlMode.acceleration:
            self.apply_acceleration_action(action, dt)
        else:
            raise Exception(f"ControlMode {self._mode} not implemented")

    def get_spaces(self):
        if self._mode == ControlMode.torque:
            return self.get_torque_spaces()
        elif self._mode == ControlMode.velocity:
            return self.get_velocity_spaces()
        elif self._mode == ControlMode.acceleration:
            return self.get_acceleration_spaces()
        else:
            raise Exception(f"ControlMode {self._mode} not implemented")

    @abstractmethod
    def update_state(self) -> None:
        """Updates the state of the robot.

        This function reads current joint position and velocities from the
        physics engine.

        """
        pass

    def sense(self, obst_ids: List[int], goal_ids: List[int]) -> None:
        """Updates the sensing of the robot's sensors."""
        self.sensor_observation = {}
        for sensor in self._sensors:
            self.sensor_observation[sensor.name()] = sensor.sense(self._robot, obst_ids, goal_ids)
            #self.sensor_observation.update(sensor.sense(self._robot, obst_ids, goal_ids))

    def get_observation(self, obst_ids: List[int], goal_ids: List[int]) -> dict:
        """Updates all observation and concatenate joint states and sensor
        observation."""
        self.update_state()
        self.sense(obst_ids, goal_ids)
        return {**self.state, **self.sensor_observation}

    def add_sensor(self, sensor: Sensor) -> int:
        """Adds sensor to the robot."""
        self._sensors.append(sensor)
        return sensor.get_observation_size()

    def sensors(self) -> list:
        return self._sensors
