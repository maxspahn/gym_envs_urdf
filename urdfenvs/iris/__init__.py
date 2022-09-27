from gym.envs.registration import register
register(
    id='iris-rotor-vel-v0',
    entry_point='urdfenvs.iris.envs:IRISRotorVelEnv'
)
