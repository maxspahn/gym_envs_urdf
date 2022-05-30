from gym.envs.registration import register
register(
    id='braitenberg-robot-vel-v0',
    entry_point='urdfenvs.braitenberg_robot.envs:BraitenbergRobotVelEnv'
)
register(
    id='braitenberg-robot-acc-v0',
    entry_point='urdfenvs.braitenberg_robot.envs:BraitenbergRobotAccEnv'
)
