from gymnasium import register
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.generic_mujoco_env import GenericMujocoEnv
register(
    id='urdf-env-v0',
    entry_point='urdfenvs.urdf_common:UrdfEnv'
)
register(
    id='generic-mujoco-env-v0',
    entry_point='urdfenvs.urdf_common:GenericMujocoEnv'
)
