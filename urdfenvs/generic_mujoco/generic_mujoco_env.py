import numpy as np
from typing import List, Optional
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import gymnasium as gym
from gymnasium import utils
import mujoco
from urdfenvs.sensors.sensor import Sensor

from urdfenvs.urdf_common.urdf_env import (
    check_observation,
    WrongObservationError,
)
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.generic_mujoco.mujoco_rendering import MujocoRenderer
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.goals.sub_goal import SubGoal

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

DEFAULT_SIZE = 480


class GenericMujocoEnv(utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    _t: float

    def __init__(
        self,
        robots: List[GenericMujocoRobot],
        obstacles: List[CollisionObstacle],
        goals: List[SubGoal],
        sensors: Optional[List[Sensor]] = None,
        render: bool = False,
        frame_skip: int = 5,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ) -> None:

        utils.EzPickle.__init__(self)
        self._xml_file = robots[0].xml_file
        self._obstacles = obstacles
        self._goals = goals
        self.add_scene()
        self._t = 0.0


        render_mode = None
        if render:
            render_mode = "human"

        self.width = width
        self.height = height


        self.frame_skip = frame_skip
        self._initialize_simulation()

        if not sensors:
            self._sensors = []
        else:
            self._sensors = sensors
            for sensor in self._sensors:
                sensor.set_data(self.data)

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, DEFAULT_CAMERA_CONFIG
        )

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def get_observation_space(self) -> gym.spaces.Dict:
        """Get observation space."""
        observation_space_as_dict= dict(gym.spaces.Dict({
            "joint_state": gym.spaces.Dict(
                {
                    "position": gym.spaces.Box(
                        low=self.joint_limits()[:, 0],
                        high=self.joint_limits()[:, 1],
                        dtype=float,
                    ),
                    "velocity": gym.spaces.Box(
                        low=np.ones_like(self.joint_limits()[:, 0])
                        * -2.0,
                        high=np.ones_like(self.joint_limits()[:, 0])
                        * 2.0,
                        dtype=float,
                    ),
                }
            )
        }))
        for sensor in self._sensors:
            observation_space_as_dict.update(
                sensor.get_observation_space(self.obstacles_dict, self.goals_dict)
            )
        observation_space = gym.spaces.Dict({'robot_0': gym.spaces.Dict(observation_space_as_dict)})
        return observation_space

    def get_action_space(self) -> np.ndarray:
        return gym.spaces.Box(
            low=self.actuator_limits()[:, 0],
            high=self.actuator_limits()[:, 1],
            dtype=float,
        )

    def joint_limits(self) -> np.ndarray:
        return self.model.jnt_range

    def actuator_limits(self) -> np.ndarray:
        return self.model.actuator_ctrlrange

    def velocity_limits(self) -> np.ndarray:
        return self.model.actuator_ctrlrange

    def _initialize_simulation(
        self,
    ):
        self.model = mujoco.MjModel.from_xml_path(self._xml_file)
        self.model.body_pos[0] = [0, 1, 1]
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)


    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    @property
    def t(self) -> float:
        return self._t

    @property
    def obstacles_dict(self) -> dict:
        return {k: v for k, v in enumerate(self._obstacles)}

    @property
    def goals_dict(self) -> dict:
        return {k: v for k, v in enumerate(self._goals)}

    def update_obstacles_position(self):
        for i, obstacle in enumerate(self._obstacles):
            self.data.mocap_pos[i] = obstacle.position(t=self.t)

    def step(self, action: np.ndarray):
        self._t += self.dt
        terminated = False
        truncated = False
        info = {}
        reward = 0
        if not self.action_space.contains(action):
            terminated = True
            info = {"action_limits": f"{action} not in {self.action_space}"}

        self.do_simulation(action, self.frame_skip)
        self.update_obstacles_position()
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        if not self.observation_space.contains(ob):
            try:
                check_observation(self.observation_space, ob)
            except WrongObservationError as e:
                self._done = True
                self._info = {"observation_limits": str(e)}
        return (
            ob,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
    ):
        if pos is not None:
            qpos = pos
        else:
            qpos = (self.joint_limits()[:, 1] + self.joint_limits()[:, 0]) / 2
        if vel is not None:
            qvel = vel
        else:
            qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        self._t = 0.0
        return self._get_obs(), {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def nu(self) -> int:
        return self.model.nu

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def add_scene(self) -> None:
        tree = ET.parse(self._xml_file)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        for obstacle in self._obstacles:
            self.add_obstacle(obstacle, worldbody)
        for sub_goal in self._goals:
            self.add_sub_goal(sub_goal, worldbody)
        self._xml_file = self._xml_file[:-4] + "_temp.xml"
        tree.write(self._xml_file)


    def add_obstacle(
        self, obst: CollisionObstacle, worldbody: ET.Element
    ) -> None:
        if obst.type() == "sphere":
            size = str(obst.size()[0])
        else:
            size = " ".join([str(i / 2) for i in obst.size()])

        geom_values = {
            "name": obst.name(),
            "type": obst.type(),
            "rgba": " ".join([str(i) for i in obst.rgba()]),
            "size": size,
        }

        body = ET.SubElement(worldbody, "body")
        body.set("name", obst.name())
        body.set("mocap", "true")
        body.set("pos", " ".join([str(i) for i in obst.position()]))

        ET.SubElement(body, "geom", geom_values)

    def add_sub_goal(self, sub_goal: SubGoal, worldbody: ET.Element) -> None:
        geom_values = {
            "name": sub_goal.name(),
            "type": "sphere",
            "rgba": "0 1 0 0.3",
            "pos": " ".join([str(i) for i in sub_goal.position()]),
            "size": str(sub_goal.epsilon()),
            "contype": str(0),
            "conaffinity": str(0),
        }
        ET.SubElement(worldbody, "geom", geom_values)

    def _get_obs(self):
        observation = {
            "joint_state": {
                "position": self.data.qpos,
                "velocity": self.data.qvel,
            }
        }
        for sensor in self._sensors:
            observation.update({sensor.name(): sensor.sense(0, self.obstacles_dict, self.goals_dict, self.t)})
        return {"robot_0": observation}

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
