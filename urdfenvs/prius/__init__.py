from gym.envs.registration import register
register(
    id='prius-vel-v0',
    entry_point='urdfenvs.prius.envs:PriusVelEnv'
)
