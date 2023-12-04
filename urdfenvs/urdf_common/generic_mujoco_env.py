import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_env import MuJocoPyEnv
from gymnasium.spaces import Box
import mujoco
import itertools

import urdfenvs


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
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)
        xml_file = kwargs["xml_file"]
        del kwargs['xml_file']
        if not os.path.exists(xml_file):
            asset_dir = urdfenvs.__path__[0] + "/assets"
            asset_xml = None
            for root, _, files in os.walk(asset_dir):
                for file in files:
                    if file == xml_file:
                        asset_xml = os.path.join(root, file)
            if asset_xml is None:
                raise Exception(f"the request xml {xml_file} can not be found")
            self._xml_file = asset_xml
        else:
            self._xml_file = xml_file

        MujocoEnv.__init__(
            self,
            self._xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        

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

    def reset_model(self, pos=np.ones(9)):
        qpos = pos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def add_obstacles(self) -> None:
        viewer = self.mujoco_renderer.viewer
        marker = {}
        marker["pos"] = np.array([1.0, 1.0, 1.0])
        marker["size"] = np.ones(3)
        #viewer.add_marker(**marker)
        viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  pos=np.array([1, 1, 0.1]),
                  label='hi')


    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:8],
                self.data.qvel.flat[:8],
            ]
        )
