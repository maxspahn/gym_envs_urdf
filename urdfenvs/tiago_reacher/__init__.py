from gym.envs.registration import register
register(
    id='tiago-reacher-vel-v0',
    entry_point='urdfenvs.tiago_reacher.envs:TiagoReacherVelEnv'
)
register(
    id='tiago-reacher-tor-v0',
    entry_point='urdfenvs.tiago_reacher.envs:TiagoReacherTorEnv'
)
register(
    id='tiago-reacher-acc-v0',
    entry_point='urdfenvs.tiago_reacher.envs:TiagoReacherAccEnv'
)
