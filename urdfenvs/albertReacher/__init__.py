from gym.envs.registration import register
register(
    id='albert-reacher-tor-v0',
    entry_point='urdfenvs.albertReacher.envs:AlbertReacherTorEnv'
)
register(
    id='albert-reacher-vel-v0',
    entry_point='urdfenvs.albertReacher.envs:AlbertReacherVelEnv'
)
register(
    id='albert-reacher-acc-v0',
    entry_point='urdfenvs.albertReacher.envs:AlbertReacherAccEnv'
)
