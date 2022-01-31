from gym.envs.registration import register
register(
    id='panda-reacher-tor-v0',
    entry_point='urdfenvs.pandaReacher.envs:PandaReacherTorEnv'
)
register(
    id='panda-reacher-vel-v0',
    entry_point='urdfenvs.pandaReacher.envs:PandaReacherVelEnv'
)
register(
    id='panda-reacher-acc-v0',
    entry_point='urdfenvs.pandaReacher.envs:PandaReacherAccEnv'
)
