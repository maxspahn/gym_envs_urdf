from gym.envs.registration import register
from urdfenvs.urdfCommon.urdf_env import UrdfEnv
register(
    id='urdf-env-v0',
    entry_point='urdfenvs.urdfCommon:UrdfEnv'
)
