from gym.envs.registration import register
from urdfenvs.urdf_common.urdf_env import UrdfEnv
register(
    id='urdf-env-v0',
    entry_point='urdfenvs.urdf_common:UrdfEnv'
)
