from gym.envs.registration import register
register(
    id='albert-reacher-tor-v0',
    entry_point='urdfenvs.albert_reacher.envs:AlbertReacherTorEnv'
)
register(
    id='albert-reacher-vel-v0',
    entry_point='urdfenvs.albert_reacher.envs:AlbertReacherVelEnv'
)
register(
    id='albert-reacher-acc-v0',
    entry_point='urdfenvs.albert_reacher.envs:AlbertReacherAccEnv'
)
