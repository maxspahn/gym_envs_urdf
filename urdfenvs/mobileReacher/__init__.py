from gym.envs.registration import register
register(
    id='mobile-reacher-tor-v0',
    entry_point='urdfenvs.mobileReacher.envs:MobileReacherTorEnv'
)
register(
    id='mobile-reacher-vel-v0',
    entry_point='urdfenvs.mobileReacher.envs:MobileReacherVelEnv'
)
register(
    id='mobile-reacher-acc-v0',
    entry_point='urdfenvs.mobileReacher.envs:MobileReacherAccEnv'
)
