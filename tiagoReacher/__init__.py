from gym.envs.registration import register
register(
    id='tiago-reacher-vel-v0',
    entry_point='tiagoReacher.envs:TiagoReacherVelEnv'
)
