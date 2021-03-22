from gym.envs.registration import register
register(
    id='panda-reacher-tor-v0',
    entry_point='pandaReacher.envs:PandaReacherTorEnv'
)
register(
    id='panda-reacher-vel-v0',
    entry_point='pandaReacher.envs:PandaReacherVelEnv'
)
