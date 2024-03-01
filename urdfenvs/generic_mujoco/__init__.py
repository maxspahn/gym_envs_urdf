from gymnasium import register
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
register(
    id='generic-mujoco-env-v0',
    entry_point='urdfenvs.generic_mujoco.generic_mujoco_env:GenericMujocoEnv'
)
