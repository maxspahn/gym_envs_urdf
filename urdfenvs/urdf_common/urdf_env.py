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
    pass

class WrongActionError(Exception):
    pass


def check_observation(obs, ob):
    for key, value in ob.items():
        if isinstance(value, dict):
            check_observation(obs[key], value)
        elif isinstance(value, np.ndarray):
            if not obs[key].contains(value):
                s = f"key: {key}: {value} not in {obs[key]}"
                if np.any(value < obs[key].low):
                    index = np.where(value < obs[key].low)[0]
                    value_at_index = value[index]
                    s += f"\nAt index {index.tolist()}: {value_at_index} < {obs[key].low[index]}"
                if np.any(value > obs[key].high):
                    index = np.where(value > obs[key].high)[0]
                    value_at_index = value[index]
                    s += f"\nAt index {index.tolist()}: {value_at_index} > {obs[key].high[index]}"

                raise WrongObservationError(s)
        else:
            raise Exception(f"Observation checking failed for key:{key} value:{value}.")


class UrdfEnv(gym.Env):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self,
        robots: List[GenericRobot],
        render: bool = False,
        dt: float = 0.01,
        observation_checking=True,
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
        self._info: dict = {}
        self._num_sub_steps: float = 20
        self._obsts: dict = {}
        self._goals: dict = {}
        self._space_set = False
        self._observation_checking = observation_checking
        if self._render:
            self._cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self._cid = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._num_sub_steps
        )
        self.plane = Plane()
        p.setGravity(0, 0, -10.0)
        self._obsts = {}
        self._goals = {}
        self.set_spaces()

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
        return (
            camera_distance,
            camera_yaw,
            camera_pitch,
            camera_target_position,
        )

    def reconfigure_camera(
        self,
        camera_distance: float,
        camera_yaw: float,
        camera_pitch: float,
        camera_target_position: tuple,
    ) -> None:
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
            logging.warning("Video recording requires rendering to be active.")

    def stop_video_recording(self) -> None:
        if self._render:
            p.stopStateLogging()

    def set_spaces(self) -> None:
        """Set observation and action space."""
        observation_space_as_dict = {}
        action_space_as_dict = {}

        for i, robot in enumerate(self._robots):
            (obs_space_robot_i, action_space_robot_i) = robot.get_spaces()
            obs_space_robot_i = dict(obs_space_robot_i)
            for sensor in robot._sensors:
                obs_space_robot_i.update(
                    sensor.get_observation_space(self._obsts, self._goals)
                )
            observation_space_as_dict[f"robot_{i}"] = gym.spaces.Dict(
                obs_space_robot_i
            )
            action_space_as_dict[f"robot_{i}"] = action_space_robot_i

        self.observation_space = gym.spaces.Dict(observation_space_as_dict)
        action_space = gym.spaces.Dict(action_space_as_dict)
        self.action_space = gym.spaces.flatten_space(action_space)

    def step(self, action):
        self._t += self.dt()
        # Feed action to the robot and get observation of robot's state

        if not self.action_space.contains(action):
            self._done = True
            self._info = {'action_limits': f"{action} not in {self.action_space}"}

        action_id = 0
        for robot in self._robots:
            action_robot = action[action_id : action_id + robot.n()]
            robot.apply_action(action_robot, self.dt())
            action_id += robot.n()

        self.update_obstacles()
        self.update_goals()
        p.stepSimulation(self._cid)
        ob = self._get_ob()

        reward = 1.0

        if self._render:
            self.render()
        return ob, reward, self._done, self._info

    def _get_ob(self) -> dict:
        """Compose the observation."""
        observation = {}
        for i, robot in enumerate(self._robots):
            obs = robot.get_observation(self._obsts, self._goals, self.t())

            observation[f"robot_{i}"] = obs
        if hasattr(self, "observation_space"):
            if (
                not self.observation_space.contains(observation)
                and self._observation_checking
            ):
                try:
                    check_observation(self.observation_space, observation)
                except WrongObservationError as e:
                    self._done = True
                    self._info = {'observation_limits': str(e)}
        return observation

    def shuffle_obstacles(self) -> dict:
        obstacle_dict = {}
        for obst_id, obst in self._obsts.items():
            obst.shuffle()
            obstacle_dict[obst.name()] = obst.dict()
        self.update_obstacles()
        return obstacle_dict

    def shuffle_goals(self) -> dict:
        goal_dict = {}
        for goal_id, goal in self._goals.items():
            goal.shuffle()
            goal_dict[goal.name()] = goal.dict()
        self.update_goals()
        return goal_dict

    def empty_scene(self) -> None:
        for goal_id  in self._goals.keys():
            p.removeBody(goal_id)
        self._goals = {}
        for obst_id in self._obsts.keys():
            p.removeBody(obst_id)
        self._obsts = {}

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
        if obst.type() == "urdf":
            obst_id = self.add_shape(obst.type(), obst.size(), urdf=obst.urdf())
        else:
            obst_id = self.add_shape(
                obst.type(),
                obst.size(),
                position=obst.position(),
                movable=obst.movable(),
            )
        self._obsts[obst_id] = obst
        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started"
            )

    def reset_obstacles(self) -> None:
        for obst_id, obstacle in self._obsts.items():
            if obstacle.type() == "urdf":
                pos = obstacle.position()
                vel = obstacle.position()
            else:
                pos = obstacle.position(t=0).tolist()
                vel = obstacle.velocity(t=0).tolist()
            ori = [0, 0, 0, 1]
            p.resetBasePositionAndOrientation(obst_id, pos, ori)
            p.resetBaseVelocity(obst_id, linearVelocity=vel)

    def reset_goals(self) -> None:
        for goal_id, goal in self._goals.items():
            pos = goal.position(t=0).tolist()
            vel = goal.velocity(t=0).tolist()
            ori = [0, 0, 0, 1]
            p.resetBasePositionAndOrientation(goal_id, pos, ori)
            p.resetBaseVelocity(goal_id, linearVelocity=vel)

    def get_obstacles(self) -> dict:
        return self._obsts

    def add_sub_goal(self, goal: SubGoal) -> int:
        rgba_color = [0.0, 1.0, 0.0, 0.3]
        visual_shape_id = p.createVisualShape(
            p.GEOM_SPHERE, rgbaColor=rgba_color, radius=goal.epsilon()
        )
        collision_shape = -1
        base_position = [
            0,
        ] * 3
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
                radius=size[0],
            )

        elif shape_type == "box":
            half_extens = [s / 2 for s in size]
            position = [position[i] - size[i] for i in range(3)]
            shape_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extens
            )
            visual_shape_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=[1.0, 0.0, 0.0, 1.0],
                specularColor=[1.0, 0.5, 0.5],
                halfExtents=half_extens,
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
                fileName=urdf, basePosition=position, globalScaling=scaling
            )
            return shape_id
        else:
            warnings.warn("Unknown shape type: {shape_type}, aborting...")
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

        Adding a sensor to the list of sensors.
        """
        for i in robot_ids:
            self._robots[i].add_sensor(sensor)

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
        if mount_positions is None:
            mount_positions = np.tile(np.zeros(3), (len(self._robots), 1))
        if mount_orientations is None:
            mount_orientations = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0]), (len(self._robots), 1)
            )
        if pos is None:
            pos = np.tile(None, len(self._robots))
        if vel is None:
            vel = np.tile(None, len(self._robots))
        if len(pos.shape) == 1 and len(self._robots) == 1:
            pos = np.tile(pos, (1, 1))
        if len(vel.shape) == 1 and len(self._robots) == 1:
            vel = np.tile(vel, (1, 1))
        for i, robot in enumerate(self._robots):
            checked_position, checked_velocity = robot.check_state(
                pos[i], vel[i]
            )
            robot.reset(
                pos=checked_position,
                vel=checked_velocity,
                mount_position=mount_positions[i],
                mount_orientation=mount_orientations[i],
            )
        self.reset_obstacles()
        self.reset_goals()
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
