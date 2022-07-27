import pybullet as p
from abc import ABC, abstractmethod
import gym
import numpy as np
from urdfpy import URDF
from urdfenvs.sensors.sensor import Sensor


class GenericRobot(ABC):
    """GenericRobot."""

    def __init__(self, n: int, urdf_file: str):
        """Constructor for generic robot.

        Parameters
        ----------

        n: int : Degrees of freedom of the robot
        urdf_file: str : Full path to urdf file
        """
        self._n: int = n
        self._urdf_file: str = urdf_file
        self._sensors = []
        self._urdf_robot = URDF.load(self._urdf_file)

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
        for i, joint in enumerate(self._urdf_robot.joints):
            if joint.name in self._joint_names:
                self._urdf_joints.append(i)
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
                        low=self._limitPos_j[0, :],
                        high=self._limitPos_j[1, :],
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

    @abstractmethod
    def update_state(self) -> None:
        """Updates the state of the robot.

        This function reads current joint position and velocities from the
        physics engine.

        """
        pass

    def update_sensing(self) -> None:
        """Updates the sensing of the robot's sensors."""
        self.sensor_observation = {}
        for sensor in self._sensors:
            self.sensor_observation[sensor.name()] = sensor.sense(self._robot)

    def get_observation(self) -> dict:
        """Updates all observation and concatenate joint states and sensor
        observation."""
        self.update_state()
        self.update_sensing()
        return {**self.state, **self.sensor_observation}

    def add_sensor(self, sensor: Sensor) -> int:
        """Adds sensor to the robot."""
        self._sensors.append(sensor)
        return sensor.get_observation_size()

    def sensors(self) -> list:
        return self._sensors
