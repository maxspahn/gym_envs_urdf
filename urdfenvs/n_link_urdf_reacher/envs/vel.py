from urdfenvs.n_link_urdf_reacher.envs.n_link_urdf_reacher_env import (
    NLinkUrdfReacherEnv,
)
from urdfenvs.generic_urdf_reacher.envs.vel import GenericUrdfReacherVelEnv

class NLinkUrdfVelReacherEnv(NLinkUrdfReacherEnv, GenericUrdfReacherVelEnv):
    pass
