import gym
import time
import numpy as np
import pybullet as p
import warnings
from abc import abstractmethod

from urdfenvs.urdfCommon.plane import Plane
from urdfenvs.sensors.sensor import Sensor
from urdfenvs.urdfCommon.generic_robot import GenericRobot


class WrongObservationError(Exception):
    """Exception when observation lays outside the defined observation space.

    This Exception is initiated when an the observation is not within the
    defined observation space. The purpose of this exception is to give
    the user better information about which specific part of the observation
    caused the problem.
    """

    def __init__(self, msg: str, observation: dict, observationSpace):
        """Constructor for error message.

        Parameters
        ----------

        msg: Default error message
        observation: Observation when mismatch occured
        observationSpace: Observation space of environment
        """
        msg_ext = self.get_wrong_observation(observation, observationSpace)
        super().__init__(msg + msg_ext)

    def get_wrong_observation(self, o: dict, os) -> str:
        """Detecting where the error occured.

        Parameters
        ----------

        o: observation
        os: observation space
        """
        msg_ext = ":\n"
        msg_ext += self.check_dict(o, os)
        return msg_ext

    def check_dict(
        self, o_dict: dict, os_dict, depth: int = 1, tabbing: str = ""
    ) -> str:
        """Checking correctness of dictonary observation.

        This methods searches for the cause for wrong observation.
        It loops over all keys in this dictonary and verifies whether
        observation and observation spaces fit together. If this is not
        the case, the concerned key is checked again. As the observation
        might have nested dictonaries, this function is called
        recursively.

        Parameters
        ----------

        o_dict: observation dictonary
        os_dict: observation space dictonary
        depth: current depth of nesting
        tabbing: tabbing for error message
        """
        msg_ext = ""
        for key in o_dict.keys():
            if not os_dict[key].contains(o_dict[key]):
                if isinstance(o_dict[key], dict):
                    msg_ext += tabbing + key + "\n"
                    msg_ext += self.check_dict(
                        o_dict[key],
                        os_dict[key],
                        depth=depth + 1,
                        tabbing=tabbing + "\t",
                    )
                else:
                    msg_ext += self.check_box(
                        o_dict[key], os_dict[key], key, tabbing
                    )
        return msg_ext

    def check_box(
        self, o_box: np.ndarray, os_box, key: str, tabbing: str
    ) -> str:
        """Checks correctness of box observation.

        This methods detects which value in the observation caused the
        error to be raised. Then it updates the error message msg.

        Parameters
        ----------

        o_box: observation box
        os_box: observation space box
        key: key of observation
        depth: current depth of nesting
        tabbing: current tabbing for error message
        """
        msg_ext = tabbing + "Error in " + key + "\n"
        for i, val in enumerate(o_box):
            if val < os_box.low[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} < {os_box.low[i]}\n"
            elif val > os_box.high[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} > {os_box.high[i]}\n"
        return msg_ext


class UrdfEnv(gym.Env):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self, robot: GenericRobot, render: bool = False, dt: float = 0.01
    ) -> None:
        """Constructor for environment.

        Variables are set and the pyhsics engine is initiated. Either with
        rendering (p.GUI) or without (p.DIRECT). Note that rendering slows
        down the simulation.

        Parameters:

        robot: Robot instance to be simulated
        render: Flag if simulator should render
        dt: Time step for pyhsics engine
        """
        self._dt: float = dt
        self._t: float = 0.0
        self._robot: GenericRobot = robot
        self._render: bool = render
        self._done: bool = False
        self._num_sub_steps: float = 20
        self._obsts: list = []
        self._goals: list = []

        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)

    def n(self) -> int:
        return self._robot.n()

    def dt(self) -> float:
        return self._dt

    def t(self) -> float:
        return self._t

    @abstractmethod
    def set_spaces(self) -> None:
        """Set observation and action space."""
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Applies a given action to the robot."""
        pass

    def step(self, action):
        self._t += self.dt()
        # Feed action to the robot and get observation of robot's state
        self.apply_action(action)
        for obst in self._obsts:
            obst.updateBulletPosition(p, t=self.t())
        for goal in self._goals:
            goal.updateBulletPosition(p, t=self.t())
        p.stepSimulation()
        ob = self._get_ob()

        reward = 1.0

        if self._render:
            self.render()
        return ob, reward, self._done, {}

    def _get_ob(self) -> dict:
        """Compose the observation."""
        observation = self._robot.get_observation()
        if not self.observation_space.contains(observation):
            err = WrongObservationError(
                "The observation does not fit the defined observation space",
                observation,
                self.observation_space,
            )
            warnings.warn(str(err))
        return observation

    def add_obstacle(self, obst) -> None:
        """Adds obstacle to the simulation environment.

        Parameters
        ----------

        obst: Obstacle from MotionPlanningEnv
        """
        # add obstacle to environment
        self._obsts.append(obst)
        obst.add2Bullet(p)

        # refresh observation space of robots sensors
        sensors = self._robot.sensors()
        cur_dict = dict(self.observation_space.spaces)
        for sensor in sensors:
            cur_dict[sensor.name()] = sensor.get_observation_space()
        self.observation_space = gym.spaces.Dict(cur_dict)

        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started"
            )

    def get_obstacles(self) -> list:
        return self._obsts

    def add_goal(self, goal) -> None:
        """Adds goal to the simulation environment.

        Parameters
        ----------

        goal: Goal from MotionPlanningGoal
        """
        self._goals.append(goal)
        goal.add2Bullet(p)

    def set_walls(self, limits: list = [[-2, -2], [2, 2]]) -> None:
        """Adds walls to the simulation environment.
        # TODO: To be removed in the future and incorporated
        into addObstacle <10-03-22, maxspahn> #

        Parameters
        ----------

        limits: Positions of walls, [[x_low, y_low], [x_high, y_high]]
        """
        col_wall_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.05, 10.0, 0.5]
        )
        p.createMultiBody(
            0,
            col_wall_id,
            10,
            [limits[0][0], 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )
        p.createMultiBody(
            0,
            col_wall_id,
            10,
            [limits[1][0], 0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )
        p.createMultiBody(
            0,
            col_wall_id,
            10,
            [0, limits[0][1], 0.0],
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        )
        p.createMultiBody(
            0,
            col_wall_id,
            10,
            [0, limits[1][1], 0.0],
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        )

    def add_sensor(self, sensor: Sensor) -> None:
        """Adds sensor to the robot.

        Adding a sensor requires an update to the observation space.
        This seems to require a conversion to dict and back to
        gym.spaces.Dict.
        """
        self._robot.add_sensor(sensor)
        cur_dict = dict(self.observation_space.spaces)
        cur_dict[sensor.name()] = sensor.get_observation_space()
        self.observation_space = gym.spaces.Dict(cur_dict)

    def check_initial_state(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters initial state of the robot and returns a valid state."""

        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            pos = np.zeros(self._robot.n())
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> dict:
        """Resets the simulation and the robot.

        Parameters
        ----------

        pos: np.ndarray: Initial joint positions of the robot
        vel: np.ndarray: Initial joint velocities of the robot
        """
        self._t = 0.0
        pos, vel = self.check_initial_state(pos, vel)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._num_sub_steps
        )
        self._robot.reset(pos=pos, vel=vel)
        self.plane = Plane()
        p.setGravity(0, 0, -10.0)
        p.stepSimulation()
        return self._robot.get_observation()

    def render(self) -> None:
        """Rendering the simulation environmemnt.

        As rendering is done rather by the self._render flag,
        only the sleep statement is called here. This speeds up
        the simulation when rendering is not desired.

        """
        time.sleep(self.dt())

    def close(self) -> None:
        p.disconnect()
