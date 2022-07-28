from urdfenvs.n_link_urdf_reacher.envs.n_link_urdf_reacher_env import (
    NLinkUrdfReacherEnv,
)
from urdfenvs.generic_urdf_reacher.envs.tor import GenericUrdfReacherTorEnv

class NLinkUrdfTorReacherEnv(NLinkUrdfReacherEnv, GenericUrdfReacherTorEnv):
    pass
