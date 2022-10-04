from gym.envs.registration import register
register(
    id='generic-urdf-reacher-tor-v0',
    entry_point='urdfenvs.generic_urdf_reacher.envs:GenericUrdfReacherTorEnv'
)
register(
    id='generic-urdf-reacher-vel-v0',
    entry_point='urdfenvs.generic_urdf_reacher.envs:GenericUrdfReacherVelEnv'
)
register(
    id='generic-urdf-reacher-acc-v0',
    entry_point='urdfenvs.generic_urdf_reacher.envs:GenericUrdfReacherAccEnv'
)
register(
    id='generic-urdf-reacher-v0',
    entry_point='urdfenvs.generic_urdf_reacher.envs:GenericUrdfReacherEnv'
)
