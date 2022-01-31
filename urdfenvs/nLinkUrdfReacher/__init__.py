from gym.envs.registration import register
register(
    id='nLink-urdf-reacher-tor-v0',
    entry_point='urdfenvs.nLinkUrdfReacher.envs:NLinkUrdfTorReacherEnv'
)
register(
    id='nLink-urdf-reacher-vel-v0',
    entry_point='urdfenvs.nLinkUrdfReacher.envs:NLinkUrdfVelReacherEnv'
)
register(
    id='nLink-urdf-reacher-acc-v0',
    entry_point='urdfenvs.nLinkUrdfReacher.envs:NLinkUrdfAccReacherEnv'
)
