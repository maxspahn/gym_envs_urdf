import gym
import time
import deprecation
import numpy as np
import pybullet as p
import warnings
from typing import List
from urdfenvs import __version__

from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.goals.goal_composition import GoalComposition

from urdfenvs.urdf_common.plane import Plane
from urdfenvs.sensors.sensor import Sensor
from urdfenvs.urdf_common.generic_robot import GenericRobot


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
        """Checking correctness of dictionary observation.

        This methods searches for the cause for wrong observation.
        It loops over all keys in this dictionary and verifies whether
        observation and observation spaces fit together. If this is not
        the case, the concerned key is checked again. As the observation
        might have nested dictionaries, this function is called
        recursively.

        Parameters
        ----------

        o_dict: observation dictionary
        os_dict: observation space dictionary
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
        tabbing: current tabbing for error message
        """
        msg_ext = tabbing + "Error in " + key + "\n"
        if isinstance(o_box, float):
            val = o_box
            if val < os_box.low[0]:
                msg_ext += f"{tabbing}\t{key}: {val} < {os_box.low[0]}\n"
            elif val > os_box.high[0]:
                msg_ext += f"{tabbing}\t{key}: {val} > {os_box.high[0]}\n"
            return msg_ext

        for i, val in enumerate(o_box):
            if val < os_box.low[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} < {os_box.low[i]}\n"
            elif val > os_box.high[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} > {os_box.high[i]}\n"
        return msg_ext

def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array


def filter_shape_dim(
    dim: np.ndarray, shape_type: str, dim_len: int, default: np.ndarray
) -> np.ndarray:
    """
    Checks and filters the dimension of a shape depending
    on the shape, warns were necessary.

    Parameters
    ----------

    dim: the dimension of the shape
    shape_type: the shape type
    dim_len: the number of dimensions should equal dim_len
    default: fallback option for dim

    """

    # check dimensions
    if isinstance(dim, np.ndarray) and np.size(dim) is dim_len:
        pass
    elif dim is None:
        dim = default
    else:
        warnings.warn(
            f"{shape_type} dimension should be of"
            "type (np.ndarray, list) with shape = ({dim_len}, )\n"
            " currently type(dim) = {type(dim)}. Aborting..."
        )
        return default
    return dim


class UrdfEnv(gym.Env):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self, robots: List[GenericRobot], flatten_observation: bool = False,
        render: bool = False, dt: float = 0.01
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
        assert len(robots) > 0
        self._dt: float = dt
        self._t: float = 0.0
        self._robots: List[GenericRobot] = robots
        self._render: bool = render
        self._done: bool = False
        self._num_sub_steps: float = 20
        self._obsts: dict = {}
        self._goals: dict = {}
        self._flatten_observation: bool = flatten_observation
        self._space_set = False
        if self._render:
            self._cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self._cid = p.connect(p.DIRECT)

    def n(self) -> int:
        return sum(self.n_per_robot())

    def n_per_robot(self) -> list:
        return [robot.n() for robot in self._robots]

    def ns_per_robot(self) -> list:
        return [robot.ns() for robot in self._robots]

    def dt(self) -> float:
        return self._dt

    def t(self) -> float:
        return self._t

    def set_spaces(self) -> None:
        """Set observation and action space."""
        self.observation_space = {}
        self.action_space = {}

        for i, robot in enumerate(self._robots):
            ( 
                obs_space,
                action_space
            ) = robot.get_spaces()

            self.observation_space[f'robot_{i}'] = obs_space
            self.action_space[f'robot_{i}'] = action_space

    def step(self, actions):
        self._t += self.dt()
        # Feed action to the robot and get observation of robot's state

        action_id = 0
        for robot in self._robots:
            action = actions[action_id:action_id+robot.n()]
            robot.apply_action(action, self.dt())
            action_id +=robot.n()

        for obst in self._obsts.values():
            obst.update_bullet_position(p, t=self.t())
        for goal in self._goals.values():
            goal.update_bullet_position(p, t=self.t())
        p.stepSimulation(self._cid)
        ob = self._get_ob()

        reward = 1.0

        if self._render:
            self.render()
        return ob, reward, self._done, {}

    def _get_ob(self) -> dict:
        """Compose the observation."""
        observation = {}
        for i, robot in enumerate(self._robots):
            obs = robot.get_observation(list(self._obsts.keys()), list(self._goals.keys()))

            observation[f'robot_{i}'] = obs

            # TODO: Make this check work for the whole observation space (not just 'joint_state'). This also breaks BicycleModel.
            if not self.observation_space[f'robot_{i}']['joint_state'].contains(observation[f'robot_{i}']['joint_state']):
                err = WrongObservationError(
                    "The observation does not fit the defined observation space",
                    observation[f'robot_{i}']['joint_state'],
                    self.observation_space[f'robot_{i}']['joint_state'],
                )
                warnings.warn(str(err))

        if self._flatten_observation:
            return flatten_observation(observation)
        else:
            return observation

    def add_obstacle(self, obst: CollisionObstacle) -> None:
        """Adds obstacle to the simulation environment.

        Parameters
        ----------

        obst: Obstacle from mpscenes
        """
        # add obstacle to environment
        obst_id = obst.add_to_bullet(p)
        self._obsts[obst_id] = obst

        # refresh observation space of robots sensors
        for i, robot in enumerate(self._robots):
            cur_dict = dict(self.observation_space[f'robot_{i}'].spaces)
            sensors = robot.sensors()
            for sensor in sensors:
                cur_dict[sensor.name()] = sensor.get_observation_space()

            self.observation_space[f'robot_{i}'] = gym.spaces.Dict(cur_dict)

        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started"
            )

    def get_obstacles(self) -> dict:
        return self._obsts

    def add_goal(self, goal: GoalComposition) -> None:
        """Adds goal to the simulation environment.

        Parameters
        ----------

        goal: Goal from mpscenes
        """
        goal_id = goal.add_to_bullet(p)
        self._goals[goal_id] = goal

    @deprecation.deprecated(deprecated_in="0.4.3",
                        current_version=__version__,
                        details="Use the explicit obstacle box from mpscenes function instead")
    def add_walls(
        self,
        dim=np.array([0.2, 8, 0.5]),
        poses_2d=None,
    ) -> None:
        """
        Adds walls to the simulation environment.

        Parameters
        ----------

        dim = [width, length, height]
        poses_2d = [[x_position, y_position, orientation], ...]
        """
        if poses_2d is None:
            poses_2d = [
                [-4, 0.1, 0],
                [4, -0.1, 0],
                [0.1, 4, 0.5 * np.pi],
                [-0.1, -4, 0.5 * np.pi],
            ]
        self.add_shapes(
            shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=poses_2d
        )

    def add_shapes(
        self,
        shape_type: str,
        dim=None,
        mass: float = 0,
        poses_2d: list = None,
        place_height=None,
    ) -> None:
        """
        Adds a shape to the simulation environment.

        Parameters
        ----------

        shape_type: str
            options are:
                "GEOM_SPHERE",
                "GEOM_BOX",
                "GEOM_CYLINDER",
                "GEOM_CAPSULE"
            .
        dim: np.ndarray or list
            dimensions for the shape, dependent on the shape_type:
                GEOM_SPHERE,    dim=[radius]
                GEOM_BOX,       dim=[width, length, height]
                GEOM_CYLINDER,  dim=[radius, length]
                GEOM_CAPSULE,   dim=[radius, length]
        mass: float
            objects mass (default = 0 : fixed shape)
        poses_2d: list
            poses where the shape should be placed. Each element
            must be of form [x_position, y_position, orientation]
        place_height: float
            z_position of the center of mass
        """
        if poses_2d is None:
            poses_2d = [[-2, 2, 0]]
        # convert list to numpy array
        if isinstance(dim, list):
            dim = np.array(dim)

        # create collisionShape
        if shape_type == "GEOM_SPHERE":
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_SPHERE", 1, default=np.array([0.5])
            )
            shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=dim[0])
            default_height = dim[0]

        elif shape_type == "GEOM_BOX":
            if dim is not None:
                dim = 0.5 * dim
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_BOX", 3, default=np.array([0.5, 0.5, 0.5])
            )
            shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
            default_height = dim[2]

        elif shape_type == "GEOM_CYLINDER":
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_CYLINDER", 2, default=np.array([0.5, 1.0])
            )
            shape_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=dim[0], height=dim[1]
            )
            default_height = 0.5 * dim[1]

        elif shape_type == "GEOM_CAPSULE":
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_CAPSULE", 2, default=np.array([0.5, 1.0])
            )
            shape_id = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=dim[0], height=dim[1]
            )
            default_height = dim[0] + 0.5 * dim[1]

        else:
            warnings.warn(
                "Unknown shape type: {shape_type}, aborting..."
            )
            return

        if place_height is None:
            place_height = default_height

        # place the shape at poses_2d
        for pose in poses_2d:
            p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=shape_id,
                baseVisualShapeIndex=-1,
                basePosition=[pose[0], pose[1], place_height],
                baseOrientation=p.getQuaternionFromEuler([0, 0, pose[2]]),
            )

        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started"
            )

    def add_sensor(self, sensor: Sensor, robot_ids: List) -> None:
        """Adds sensor to the robot.

        Adding a sensor requires an update to the observation space.
        This seems to require a conversion to dict and back to
        gym.spaces.Dict.
        """
        for i in robot_ids:
            self._robots[i].add_sensor(sensor)
            if self.observation_space:
                cur_dict = dict(self.observation_space[f'robot_{i}'].spaces)
            else:
                raise KeyError(f"Observation space for robot {i} has not been created. Add sensor after reset.")
            cur_dict[sensor.name()] = sensor.get_observation_space()
            self.observation_space[f'robot_{i}'] = gym.spaces.Dict(cur_dict)

    def reset(
            self,
            pos: np.ndarray = None,
            vel: np.ndarray = None,
            mount_positions: np.ndarray = None,
            mount_orientations: np.ndarray = None,
        ) -> dict:
        """Resets the simulation and the robot.

        Parameters
        ----------

        pos: np.ndarray:
            Initial joint positions of the robots
        vel: np.ndarray: 
            Initial joint velocities of the robots
        mount_position: np.ndarray:
            Mounting position for the robots  
            This is ignored for mobile robots
        mount_orientation: np.ndarray:
            Mounting position for the robots  
            This is ignored for mobile robots
        """
        self._t = 0.0
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._num_sub_steps
        )
        if mount_positions is None:
            mount_positions = np.tile(np.zeros(3), (len(self._robots), 1))
        if mount_orientations is None:
            mount_orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(self._robots), 1))
        if pos is None:
            pos = np.tile(None, len(self._robots))
        if vel is None:
            vel = np.tile(None, len(self._robots))
        if len(pos.shape) == 1 and len(self._robots) == 1:
            pos = np.tile(pos, (1, 1))
        if len(vel.shape) == 1 and len(self._robots) == 1:
            vel = np.tile(vel, (1, 1))
        for i, robot in enumerate(self._robots):
            checked_position, checked_velocity= robot.check_state(pos[i], vel[i])
            robot.reset(
                pos=checked_position,
                vel=checked_velocity,
                mount_position=mount_positions[i],
                mount_orientation=mount_orientations[i],
            )
        if not self._space_set:
            self.set_spaces()
            self._space_set = True
        self.plane = Plane()
        p.setGravity(0, 0, -10.0)
        p.stepSimulation()
        return self._get_ob()

    def render(self) -> None:
        """Rendering the simulation environment.

        As rendering is done rather by the self._render flag,
        only the sleep statement is called here. This speeds up
        the simulation when rendering is not desired.

        """
        time.sleep(self.dt())

    def close(self) -> None:
        p.disconnect(self._cid)
