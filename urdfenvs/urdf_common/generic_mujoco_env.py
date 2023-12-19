import numpy as np
from typing import List
import os
import xml.etree.ElementTree as ET

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_env import MuJocoPyEnv
from gymnasium.spaces import Box
import mujoco
import itertools

import urdfenvs
from urdfenvs.urdf_common.generic_mujoco_robot import GenericMujocoRobot
from mpscenes.obstacles.collision_obstacle import CollisionObstacle


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class GenericMujocoEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        robots: List[GenericMujocoRobot],
        obstacles: List[CollisionObstacle],
        render: bool = False,
    ) -> None:
        utils.EzPickle.__init__(self)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)
        self._xml_file = robots[0].xml_file
        self._obstacles = obstacles
        self.add_obstacles()
        render_mode = None
        if render:
            render_mode = "human"
        MujocoEnv.__init__(
            self,
            "",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
        )



    def _initialize_simulation(
        self,
    ):
        model = mujoco.MjModel.from_xml_path(self._xml_file)
        model.body_pos[0] = [0, 1, 1]
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

        

    def step(self, a):
        reward = 0

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            {},
        )

    def reset(self, pos=np.ones(9)):
        qpos = pos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()


    def add_obstacles(self) -> None:
        tree = ET.parse(self._xml_file)
        worldbody = tree.getroot().find('worldbody')
        for obstacle in self._obstacles:
            self.add_obstacle(obstacle, worldbody)
        self._xml_file = self._xml_file[:-4]+'_temp.xml'
        tree.write(self._xml_file)

    def add_obstacle(self, obst: CollisionObstacle, worldbody: ET.Element) -> None:
        geom_values = {
            'name': obst.name(),
            'type': obst.type(),
            'rgba': " ".join([str(i) for i in obst.rgba()]),
            'pos': " ".join([str(i) for i in obst.position()]),
            'size': " ".join([str(i/2) for i in obst.size()]),
        }
        ET.SubElement(worldbody, 'geom', geom_values)


    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:8],
                self.data.qvel.flat[:8],
            ]
        )
