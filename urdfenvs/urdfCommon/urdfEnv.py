import gym
import time
import numpy as np
import pybullet as p
import warnings
from abc import abstractmethod
from urdfenvs.urdfCommon.plane import Plane
from urdfenvs.sensors.sensor import Sensor

from urdfenvs.urdfCommon.genericRobot import GenericRobot


class WrongObservationError(Exception):
    """Exception when observation lays outside the defined observation space.

    This Exception is initiated when an the observation is not within the
    defined observation space. The purpose of this exception is to give
    the user better information about which specif part of the observation
    caused the problem.
    """

    def __init__(self, msg, observation, observationSpace):
        """Constructor for error message.

        :param msg: Default error message
        :param observation: Observation when mismatch occured
        :param observationSpace: Observation space of environment
        """
        msgExt = self.getWrongObservation(observation, observationSpace)
        super().__init__(msg + msgExt)

    def getWrongObservation(self, o, os):
        """Detecting where the error occured.

        :param o: observation
        :param os: observation space
        """
        msgExt = ":\n"
        msgExt += self.checkDict(o, os)
        return msgExt

    def checkDict(self, o_dict, os_dict, depth=1, tabbing=""):
        """Checking correctness of dictonary observation.

        This methods searches for the cause for wrong observation.
        It loops over all keys in this dictonary and verifies whether
        observation and observation spaces fit together. If this is not
        the case, the concerned key is checked again. As the observation
        might have nested dictonaries, this function is called
        recursively.

        :param o_dict: observation dictonary
        :param os_dict: observation space dictonary
        :param depth: current depth of nesting
        :param tabbing: tabbing for error message
        """
        msgExt = ""
        for key in o_dict.keys():
            if not os_dict[key].contains(o_dict[key]):
                if isinstance(o_dict[key], dict):
                    msgExt += tabbing + key + "\n"
                    msgExt += self.checkDict(
                        o_dict[key],
                        os_dict[key],
                        depth=depth + 1,
                        tabbing=tabbing + "\t",
                    )
                else:
                    msgExt += self.checkBox(
                        o_dict[key], os_dict[key], key, depth, tabbing
                    )
        return msgExt

    def checkBox(self, o_box, os_box, key, depth, tabbing):
        """Checks correctness of box observation.

        This methods detects which value in the observation caused the
        error to be raised. Then it updates the error message msg.

        :param o_box: observation box
        :param os_box: observation space box
        :param key: key of observation
        :param depth: current depth of nesting
        :param tabbing: current tabbing for error message
        """
        msgExt = tabbing + "Error in " + key + "\n"
        for i, val in enumerate(o_box):
            if val < os_box.low[i]:
                msgExt += f"{tabbing}\t{key}[{i}]: {val} < {os_box.low[i]}\n"
            elif val > os_box.high[i]:
                msgExt += f"{tabbing}\t{key}[{i}]: {val} > {os_box.high[i]}\n"
        return msgExt


class UrdfEnv(gym.Env):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self, robot: GenericRobot, render: bool = False, dt: float = 0.01
    ) -> None:
        """Constructor for environment.

        Variables are set and the pyhsics engine is initiated. Either with
        rendering (p.GUI) or without (p.DIRECT). Note that rendering slows
        down the simulation.

        :param robot: Robot instance to be simulated
        :param render: Flag if simulator should render
        :param dt: Time step for pyhsics engine
        """
        self._dt: float = dt
        self._t: float = 0.0
        self.robot: GenericRobot = robot
        self._render: bool = render
        self.done: bool = False
        self._numSubSteps: float = 20
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
        return self.robot.n()

    def dt(self) -> float:
        return self._dt

    def t(self) -> float:
        return self._t

    @abstractmethod
    def setSpaces(self) -> None:
        """Set observation and action space."""
        pass

    @abstractmethod
    def applyAction(self, action: np.ndarray) -> None:
        """Applies a given action to the robot."""
        pass

    def step(self, action):
        self._t += self.dt()
        # Feed action to the robot and get observation of robot's state
        self.applyAction(action)
        for obst in self._obsts:
            obst.updateBulletPosition(p, t=self.t())
        for goal in self._goals:
            goal.updateBulletPosition(p, t=self.t())
        p.stepSimulation()
        ob = self._get_ob()

        reward = 1.0

        if self._render:
            self.render()
        return ob, reward, self.done, {}

    def _get_ob(self) -> dict:
        """Compose the observation."""
        observation = self.robot.get_observation()
        if not self.observation_space.contains(observation):
            err = WrongObservationError(
                "The observation does not fit the defined observation space",
                observation,
                self.observation_space,
            )
            warnings.warn(str(err))
        return observation

    def addObstacle(self, obst) -> None:
        """Adds obstacle to the simulation environment.

        :param obst: Obstacle from MotionPlanningEnv
        """
        # add obstacle to environment
        self._obsts.append(obst)
        obst.add2Bullet(p)

        # refresh observation space of robots sensors
        sensors = self.robot.sensors()
        curDict = dict(self.observation_space.spaces)
        for sensor in sensors:
            curDict[sensor.name()] = sensor.getObservationSpace()
        self.observation_space = gym.spaces.Dict(curDict)

        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started")

    def getObstacles(self) -> list:
        return self._obsts

    def addGoal(self, goal) -> None:
        """Adds goal to the simulation environment.

        :param goal: Goal from MotionPlanningGoal
        """
        self._goals.append(goal)
        goal.add2Bullet(p)

    def setWalls(self, limits: list = [[-2, -2], [2, 2]]) -> None:
        """Adds walls to the simulation environment.
        # TODO: To be removed in the future and incorporated
        into addObstacle <10-03-22, maxspahn> #

        :param limits: Positions of walls, [[x_low, y_low], [x_high, y_high]]
        """
        colwallId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.05, 10.0, 0.5])
        p.createMultiBody(
            0,
            colwallId,
            10,
            [limits[0][0], 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )
        p.createMultiBody(
            0,
            colwallId,
            10,
            [limits[1][0], 0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )
        p.createMultiBody(
            0,
            colwallId,
            10,
            [0, limits[0][1], 0.0],
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        )
        p.createMultiBody(
            0,
            colwallId,
            10,
            [0, limits[1][1], 0.0],
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        )

    def addSensor(self, sensor: Sensor) -> None:
        """Adds sensor to the robot.

        Adding a sensor requires an update to the observation space.
        This seems to require a conversion to dict and back to
        gym.spaces.Dict.
        """
        self.robot.addSensor(sensor)
        curDict = dict(self.observation_space.spaces)
        curDict[sensor.name()] = sensor.getObservationSpace()
        self.observation_space = gym.spaces.Dict(curDict)

    def checkInitialState(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters initial state of the robot and returns a valid state."""

        if not isinstance(pos, np.ndarray) or not pos.size == self.robot.n():
            pos = np.zeros(self.robot.n())
        if not isinstance(vel, np.ndarray) or not vel.size == self.robot.n():
            vel = np.zeros(self.robot.n())
        return pos, vel

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> dict:
        """Resets the simulation and the robot.

        :param pos: Initial joint positions of the robot
        :param vel: Initial joint velocities of the robot
        """
        self._t = 0.0
        pos, vel = self.checkInitialState(pos, vel)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.robot.reset(pos=pos, vel=vel)
        self.plane = Plane()
        p.setGravity(0, 0, -10.0)
        p.stepSimulation()
        return self.robot.get_observation()

    def render(self) -> None:
        """Rendering the simulation environmemnt.

        As rendering is done rather by the self._render flag,
        only the sleep statement is called here. This speeds up
        the simulation when rendering is not desired.

        """
        time.sleep(self.dt())

    def close(self) -> None:
        p.disconnect()
