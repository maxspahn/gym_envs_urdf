import gym
import time
import numpy as np
import pybullet as p
import warnings
import logging
from typing import List, Union

from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.goals.sub_goal import SubGoal

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

    def get_camera_configuration(self) -> tuple:
        full_camera_configuration = p.getDebugVisualizerCamera()
        camera_yaw = full_camera_configuration[8]
        camera_pitch = full_camera_configuration[9]
        camera_distance = full_camera_configuration[10]
        camera_target_position = full_camera_configuration[11]
        return (camera_distance, camera_yaw, camera_pitch, camera_target_position)

    def reconfigure_camera(
            self,
            camera_distance: float,
            camera_yaw: float,
            camera_pitch: float,
            camera_target_position: tuple) -> None:
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target_position,
        )

    def start_video_recording(self, file_name: str) -> None:
        if self._render:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file_name)
        else:
            logging.warning(
                "Video recording requires rendering to be active."
            )

    def stop_video_recording(self) -> None:
        if self._render:
            p.stopStateLogging()

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

        self.update_obstacles()
        self.update_goals()
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
            obs = robot.get_observation(self._obsts, self._goals, self.t())

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

    def update_obstacles(self):
        for obst_id, obst in self._obsts.items():
            if obst.movable():
                continue
            try:
                pos = obst.position(t=self.t()).tolist()
                vel = obst.velocity(t=self.t()).tolist()
                ori = [0, 0, 0, 1]
                p.resetBasePositionAndOrientation(obst_id, pos, ori)
                p.resetBaseVelocity(obst_id, linearVelocity=vel)
            except Exception:
                continue

    def update_goals(self):
        for goal_id, goal in self._goals.items():
            try:
                pos = goal.position(t=self.t()).tolist()
                vel = goal.velocity(t=self.t()).tolist()
                ori = [0, 0, 0, 1]
                p.resetBasePositionAndOrientation(goal_id, pos, ori)
                p.resetBaseVelocity(goal_id, linearVelocity=vel)
            except Exception:
                continue

    def add_obstacle(self, obst: CollisionObstacle) -> None:
        """Adds obstacle to the simulation environment.

        Parameters
        ----------

        obst: Obstacle from mpscenes
        """
        # add obstacle to environment
        if obst.type() == 'urdf':
            obst_id = self.add_shape(obst.type(), obst.size(), urdf=obst.urdf())
        else:
            obst_id = self.add_shape(
                obst.type(),
                obst.size(),
                position=obst.position(),
                movable=obst.movable(),
            )
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

    def add_sub_goal(self, goal: SubGoal) -> int:
        rgba_color = [0.0, 1.0, 0.0, 0.3]
        visual_shape_id = p.createVisualShape(
            p.GEOM_SPHERE, rgbaColor=rgba_color, radius=goal.epsilon()
        )
        collision_shape = -1
        base_position = [0, ] * 3
        for index in range(3):
            if index in goal.indices():
                base_position[index] = goal.position()[index]

        base_orientation = [0, 0, 0, 1]

        assert isinstance(base_position, list)
        assert isinstance(base_orientation, list)
        bullet_id = p.createMultiBody(
            0,
            collision_shape,
            visual_shape_id,
            base_position,
            base_orientation,
        )
        return bullet_id

    def add_goal(self, goal: Union[GoalComposition, SubGoal]) -> None:
        """Adds goal to the simulation environment.

        Parameters
        ----------

        goal: Goal from mpscenes
        """
        if isinstance(goal, GoalComposition):
            for sub_goal in goal.sub_goals():
                goal_id = self.add_sub_goal(sub_goal)
                self._goals[goal_id] = goal
        else:
            goal_id = self.add_sub_goal(goal)
            self._goals[goal_id] = goal

    def add_shape(
        self,
        shape_type: str,
        size: list,
        movable: bool = False,
        orientation: list = (0, 0, 0, 1),
        position: list = (0, 0, 1),
        scaling: float = 1.0,
        urdf: str = None,
    ) -> int:

        mass = float(movable)
        if shape_type in ["sphere", "splineSphere", "analyticSphere"]:
            shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=size[0])
            visual_shape_id = p.createVisualShape(
                p.GEOM_SPHERE,
                rgbaColor=[1.0, 0.0, 0.0, 1.0],
                specularColor=[1.0, 0.5, 0.5],
                radius=size[0]
            )

        elif shape_type == "box":
            half_extens = [s/2 for s in size]
            position = [position[i] - size[i] for i in range(3)]
            shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extens)
            visual_shape_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=[1.0, 0.0, 0.0, 1.0],
                specularColor=[1.0, 0.5, 0.5],
                halfExtents=half_extens
            )

        elif shape_type == "cylinder":
            shape_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size[0], height=size[1]
            )

        elif shape_type == "capsule":
            shape_id = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=size[0], height=size[1]
            )
            visual_shape_id = p.createVisualShape(
                p.GEOM_CAPSULE,
                rgbaColor=[1.0, 0.0, 0.0, 1.0],
                specularColor=[1.0, 0.5, 0.5],
                radius=size[0],
                height=size[1],
            )
        elif shape_type == "urdf":
            shape_id = p.loadURDF(
                fileName=urdf,
                basePosition=position,
                globalScaling=scaling
            )
            return shape_id
        else:
            warnings.warn(
                "Unknown shape type: {shape_type}, aborting..."
            )
            return -1
        bullet_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=orientation,
        )
        return bullet_id

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
        self._obsts = {}
        self._goals = {}
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
