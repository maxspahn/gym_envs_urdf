from gym.envs.registration import register
register(
    id='generic-reacher-tor-v0',
    entry_point='urdfenvs.generic_reacher.envs:GenericReacherTorEnv'
)
register(
    id='generic-reacher-vel-v0',
    entry_point='urdfenvs.generic_reacher.envs:GenericReacherVelEnv'
)
register(
    id='generic-reacher-acc-v0',
    entry_point='urdfenvs.generic_reacher.envs:GenericReacherAccEnv'
)
