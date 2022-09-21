from gym.envs.registration import register
register(
    id='iris-rotor-v0',
    entry_point='urdfenvs.iris.envs:IRISRotorSpeedEnv'
)
