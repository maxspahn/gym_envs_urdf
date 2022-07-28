from urdfenvs.panda_reacher.envs.panda_reacher_env import PandaReacherEnv
from urdfenvs.generic_urdf_reacher.envs.vel import GenericUrdfReacherVelEnv

class PandaReacherVelEnv(PandaReacherEnv, GenericUrdfReacherVelEnv):
    pass
