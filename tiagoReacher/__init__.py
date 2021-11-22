from gym.envs.registration import register
register(
    id='tiago-reacher-vel-v0',
    entry_point='tiagoReacher.envs:TiagoReacherVelEnv'
)
register(
    id='tiago-reacher-tor-v0',
    entry_point='tiagoReacher.envs:TiagoReacherTorEnv'
)
register(
    id='tiago-reacher-acc-v0',
    entry_point='tiagoReacher.envs:TiagoReacherAccEnv'
)
