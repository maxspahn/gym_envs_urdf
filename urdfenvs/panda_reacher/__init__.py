from gym.envs.registration import register
register(
    id='panda-reacher-tor-v0',
    entry_point='urdfenvs.panda_reacher.envs:PandaReacherTorEnv'
)
register(
    id='panda-reacher-vel-v0',
    entry_point='urdfenvs.panda_reacher.envs:PandaReacherVelEnv'
)
register(
    id='panda-reacher-acc-v0',
    entry_point='urdfenvs.panda_reacher.envs:PandaReacherAccEnv'
)
