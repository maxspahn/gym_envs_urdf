import logging
import time
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, Type, Union

import dill
import gymnasium as gym
import numpy as np
import pybullet
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.goals.sub_goal import SubGoal
from mpscenes.obstacles.collision_obstacle import CollisionObstacle

from urdfenvs.sensors.sensor import Sensor
from urdfenvs.urdf_common.generic_robot import GenericRobot
from urdfenvs.urdf_common.helpers import (
    WrongObservationError,
    check_observation,
    get_transformation_matrix,
    matrix_to_quaternion,
)
from urdfenvs.urdf_common.plane import Plane
from urdfenvs.urdf_common.pybullet_helpers import add_shape
from urdfenvs.urdf_common.reward import Reward


class UrdfEnv(gym.Env):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self,
        robots: List[type(GenericRobot)],
        render: bool = False,
        enforce_real_time: Optional[bool] = None,
        dt: float = 0.01,
        num_sub_steps: int = 20,
        observation_checking=True,
    ) -> None:
        """Constructor for environment.

        Variables are set and the pyhsics engine is initiated. Either with
        rendering (pybullet.GUI) or without (pybullet.DIRECT). Note that rendering slows
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
        self._enforce_real_time: bool = (
            render if enforce_real_time is None else enforce_real_time
        )
        self._done: bool = False
        self._info: dict = {}
        self._num_sub_steps: float = num_sub_steps
        self._obsts: dict = {}
        self._collision_links: dict = {}
        self._collision_links_poses: dict = {}
        self._visualizations: dict = {}
        self._goals: dict = {}
        self._space_set = False
        self._observation_checking = observation_checking
        self._reward_calculator = None
        self.sensors = []
        self.connect_physics_engine()
        self._obsts = {}
        self._collision_links = {}
        self._goals = {}
        self.set_spaces()

    def connect_physics_engine(self):
        if self._render:
            self._cid = pybullet.connect(pybullet.GUI)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self._cid = pybullet.connect(pybullet.DIRECT)
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._num_sub_steps
        )
        self.plane = Plane()
        pybullet.setGravity(0, 0, -10.0)

    def load_environment(self):
        old_obsts = deepcopy(list(self._obsts.values()))
        self._obsts = {}
        for obst in old_obsts:
            self.add_obstacle(obst)
        old_goals = deepcopy(list(self._goals.values()))
        self._goals = {}
        for goal in old_goals:
            self.add_goal(goal)
        # TODO load collision links

    @classmethod
    def load(
        cls: Type["UrdfEnv"], file_name: str, render: bool = False
    ) -> "UrdfEnv":
        with open(file_name, "rb") as f:
            env: "UrdfEnv" = dill.load(f)
        env._render = render
        env.connect_physics_engine()
        env.reset()
        env.load_environment()
        return env

    def set_reward_calculator(self, reward_calculator: Reward) -> None:
        self._reward_calculator = reward_calculator

    def n(self) -> int:
        return sum(self.n_per_robot())

    def n_per_robot(self) -> list:
        return [robot.n() for robot in self._robots]

    def ns_per_robot(self) -> list:
        return [robot.ns() for robot in self._robots]

    @property
    def dt(self) -> float:
        return self._dt

    def t(self) -> float:
        return self._t

    def get_camera_configuration(self) -> tuple:
        full_camera_configuration = pybullet.getDebugVisualizerCamera()
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
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target_position,
        )

    def start_video_recording(self, file_name: str) -> None:
        if self._render:
            pybullet.startStateLogging(
                pybullet.STATE_LOGGING_VIDEO_MP4, file_name
            )
        else:
            logging.warning("Video recording requires rendering to be active.")

    def stop_video_recording(self) -> None:
        if self._render:
            pybullet.stopStateLogging()

    def set_spaces(self) -> None:
        """Set observation and action space."""
        observation_space_as_dict = {}
        action_space_as_dict = {}

        for i, robot in enumerate(self._robots):
            (obs_space_robot_i, action_space_robot_i) = robot.get_spaces()
            obs_space_robot_i = dict(obs_space_robot_i)
            for sensor in robot._sensors:

                self.sensors.append(
                    sensor
                )  # Add the sensor to the list of sensors.

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

    def step(self, action: np.ndarray):
        step_start = time.perf_counter()
        self._t += self.dt
        # Feed action to the robot and get observation of robot's state

        if not self.action_space.contains(action):
            self._done = True
            self._info = {
                "action_limits": f"{action} not in {self.action_space}"
            }

        action_id = 0
        for robot in self._robots:
            action_robot = action[action_id : action_id + robot.n()]
            robot.apply_action(action_robot, self.dt)
            action_id += robot.n()

        self.update_obstacles()
        self.update_goals()
        self.update_collision_links()

        pybullet.stepSimulation(self._cid)
        for robot_id, robot in enumerate(self._robots):
            contacts = pybullet.getContactPoints(robot._robot)
            for contact_info in contacts:
                body_b = contact_info[2]
                if body_b in self._obsts:
                    message = (
                        f"Collision occured at {round(self.t(), 2)} "
                        f"between robot {robot_id} and obstacle "
                        f"with id {body_b}"
                    )
                    self._info = {"Collision": message}
                    self._done = True
                    break
        ob = self._get_ob()

        # Calculate the reward.
        # If there is no reward object, then the reward is 1.0.
        if self._reward_calculator is not None:
            reward = self._reward_calculator.calculate_reward(ob)
        else:
            reward = 1.0

        if self._render:
            self.render()

        terminated = self._done
        truncated = self._get_truncated()
        step_end = time.perf_counter()
        step_time = step_end - step_start
        if self._enforce_real_time:
            sleep_time = max(0.0, self.dt - step_time)
            time.sleep(sleep_time)
        step_final_end = time.perf_counter()
        total_step_time = step_final_end - step_start
        real_time_factor = self.dt / total_step_time
        logging.info(f"Real time factor {real_time_factor}")
        return ob, reward, terminated, truncated, self._info

    def _get_truncated(self) -> bool:
        # TODO: Implement Truncated.
        return False

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
                    self._info = {"observation_limits": str(e)}
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
        for goal_id in self._goals:
            pybullet.removeBody(goal_id)
        self._goals = {}
        for obst_id in self._obsts:
            pybullet.removeBody(obst_id)
        self._obsts = {}

    def update_obstacles(self):
        for obst_id, obst in self._obsts.items():
            if obst.movable():
                continue
            try:
                pos = obst.position(t=self.t()).tolist()
                vel = obst.velocity(t=self.t()).tolist()
                ori = obst.orientation(t=self.t()).tolist()
                ori = ori[1:] + ori[:1]
                pybullet.resetBasePositionAndOrientation(obst_id, pos, ori)
                pybullet.resetBaseVelocity(obst_id, linearVelocity=vel)
            except Exception:
                continue

    def update_collision_links(self) -> None:
        for visual_shape_id, info in self._collision_links.items():
            link_state = pybullet.getLinkState(info[0], info[1])
            link_position = link_state[0]
            link_ori = np.array(link_state[1])
            transformation_matrix = get_transformation_matrix(
                link_ori, link_position
            )
            total_transformation = np.dot(transformation_matrix, info[2])
            self._collision_links_poses[f"{info[3]}_{info[1]}_{info[4]}"] = (
                total_transformation
            )
            translation, rotation = matrix_to_quaternion(
                total_transformation, ordering="xyzw"
            )
            pybullet.resetBasePositionAndOrientation(
                visual_shape_id, translation, rotation
            )

    def update_visualizations(self, positions) -> None:
        for i, (visual_shape_id, info) in enumerate(
            self._visualizations.items()
        ):
            position = positions[i]
            rotation = [1, 0, 0, 0]
            pybullet.resetBasePositionAndOrientation(
                visual_shape_id, position, rotation
            )

    def collision_links_poses(self, position_only: bool = False) -> dict:
        if position_only:
            result_dict = {}
            for key, value in self._collision_links_poses.items():
                if value is None:
                    result_dict[key] = None
                elif isinstance(value, np.ndarray):
                    result_dict[key] = value[0:3, 3]
            return result_dict
        return self._collision_links_poses

    def update_goals(self):
        for goal_id, goal in self._goals.items():
            try:
                pos = goal.position(t=self.t()).tolist()
                vel = goal.velocity(t=self.t()).tolist()
                ori = [0, 0, 0, 1]
                pybullet.resetBasePositionAndOrientation(goal_id, pos, ori)
                pybullet.resetBaseVelocity(goal_id, linearVelocity=vel)
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
            obst_id = add_shape(obst.type(), obst.size(), urdf=obst.urdf())
        else:
            obst_id = add_shape(
                obst.type(),
                obst.size(),
                obst.rgba().tolist(),
                position=obst.position().tolist(),
                orientation=obst.orientation().tolist(),
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
                ori = obstacle.orientation(t=0).tolist()
            ori = ori[1:] + ori[:1]
            pybullet.resetBasePositionAndOrientation(obst_id, pos, ori)
            pybullet.resetBaseVelocity(obst_id, linearVelocity=vel)

    def reset_goals(self) -> None:
        for goal_id, goal in self._goals.items():
            pos = goal.position(t=0).tolist()
            vel = goal.velocity(t=0).tolist()
            ori = [0, 0, 0, 1]
            pybullet.resetBasePositionAndOrientation(goal_id, pos, ori)
            pybullet.resetBaseVelocity(goal_id, linearVelocity=vel)

    def get_obstacles(self) -> dict:
        return self._obsts

    def add_visualization(
        self,
        shape_type: str = "cylinder",
        size: Optional[List[float]] = None,
        rgba_color: Optional[np.ndarray] = [1.0, 1.0, 0.0, 0.3],
    ) -> int:

        length = [1.0]
        if size is None:
            size = [1.0, 0.2]

        bullet_id = add_shape(
            shape_type,
            size,
            rgba_color,
            with_collision_shape=False,
        )

        self._visualizations[bullet_id] = bullet_id

        return bullet_id

    def add_collision_link(
        self,
        robot_index: int = 0,
        link_index: Union[int, str] = 0,
        sphere_on_link_index: int = 0,
        shape_type: str = "sphere",
        size: Optional[List[float]] = None,
        link_transformation: Optional[np.ndarray] = None,
    ) -> int:
        if size is None:
            size = [1.0]
        if link_transformation is None:
            link_transformation = np.identity(4)
        rgba_color = [1.0, 1.0, 0.0, 0.3]
        bullet_id = add_shape(
            shape_type,
            size,
            rgba_color,
            with_collision_shape=False,
        )
        if isinstance(link_index, str):
            link_index = self._robots[robot_index]._link_names.index(link_index)
        self._collision_links[bullet_id] = (
            self._robots[robot_index]._robot,
            link_index,
            link_transformation,
        )
        self._collision_links_poses[
            f"{robot_index}_{link_index}_{sphere_on_link_index}"
        ] = None
        self._collision_links[bullet_id] = (
            self._robots[robot_index]._robot,
            link_index,
            link_transformation,
            robot_index,
            sphere_on_link_index,
        )
        return bullet_id

    def add_debug_shape(
        self,
        position: Tuple[float],
        orientation: Tuple[float],
        shape_type: str = "sphere",
        size: Optional[List[float]] = None,
        rgba_color: Optional[List[float]] = None,
    ) -> None:
        if size is None:
            size = [1.0]
        add_shape(
            shape_type,
            size,
            rgba_color,
            with_collision_shape=False,
            orientation=orientation,
            position=position,
        )

    def add_sub_goal(self, goal: SubGoal) -> int:
        rgba_color = [0.0, 1.0, 0.0, 0.3]
        visual_shape_id = pybullet.createVisualShape(
            pybullet.GEOM_SPHERE, rgbaColor=rgba_color, radius=goal.epsilon()
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
        bullet_id = pybullet.createMultiBody(
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

    def add_sensor(self, sensor: Sensor, robot_ids: List) -> None:
        """Adds sensor to the robot.

        Adding a sensor to the list of sensors.
        """
        for i in robot_ids:
            self._robots[i].add_sensor(sensor)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        mount_positions: Optional[np.ndarray] = None,
        mount_orientations: Optional[np.ndarray] = None,
    ) -> tuple:
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
        super().reset(seed=seed, options=options)
        self._t = 0.0
        if mount_positions is None:
            mount_positions = np.tile(np.zeros(3), (len(self._robots), 1))
        self.mount_positions = mount_positions
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
            if isinstance(pos[i], np.ndarray) and pos[i][0] is not None:
                initial_position_i = pos[i][np.isfinite(pos[i])]
            else:
                initial_position_i = pos[i]
            checked_position, checked_velocity = robot.check_state(
                initial_position_i, vel[i]
            )
            robot.reset(
                pos=checked_position,
                vel=checked_velocity,
                mount_position=mount_positions[i],
                mount_orientation=mount_orientations[i],
            )
        self.reset_obstacles()
        self.reset_goals()
        return self._get_ob(), self._info

    def render(self) -> None:
        """Rendering the simulation environment.

        As rendering is done rather by the self._render flag,
        only the sleep statement is called here. This speeds up
        the simulation when rendering is not desired.

        """

    def close(self) -> None:
        pybullet.disconnect(self._cid)

    def dump(self, file_name: str) -> None:
        with open(file_name, "wb") as f:
            dill.dump(self, f)
