from gym.envs.registration import register
register(
    id='nLink-urdf-reacher-tor-v0',
    entry_point='urdfenvs.n_link_urdf_reacher.envs:NLinkUrdfTorReacherEnv'
)
register(
    id='nLink-urdf-reacher-vel-v0',
    entry_point='urdfenvs.n_link_urdf_reacher.envs:NLinkUrdfVelReacherEnv'
)
register(
    id='nLink-urdf-reacher-acc-v0',
    entry_point='urdfenvs.n_link_urdf_reacher.envs:NLinkUrdfAccReacherEnv'
)
