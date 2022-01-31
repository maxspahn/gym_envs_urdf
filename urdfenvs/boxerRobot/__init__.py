from gym.envs.registration import register
register(
    id='boxer-robot-vel-v0',
    entry_point='boxerRobot.envs:BoxerRobotVelEnv'
)
register(
    id='boxer-robot-acc-v0',
    entry_point='boxerRobot.envs:BoxerRobotAccEnv'
)
