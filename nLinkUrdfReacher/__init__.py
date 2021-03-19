from gym.envs.registration import register
register(
    id='nLink-urdf-reacher-tor-v0',
    entry_point='nLinkUrdfReacher.envs:NLinkUrdfTorReacherEnv'
)
