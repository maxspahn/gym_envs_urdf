from gym.envs.registration import register
register(
    id='iris-vel-v0',
    entry_point='urdfenvs.iris.envs:IRISVelEnv'
)
