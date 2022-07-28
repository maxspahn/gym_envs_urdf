from urdfenvs.n_link_urdf_reacher.envs.n_link_urdf_reacher_env import (
    NLinkUrdfReacherEnv,
)
from urdfenvs.generic_urdf_reacher.envs.acc import GenericUrdfReacherAccEnv

class NLinkUrdfAccReacherEnv(NLinkUrdfReacherEnv, GenericUrdfReacherAccEnv):
    pass

