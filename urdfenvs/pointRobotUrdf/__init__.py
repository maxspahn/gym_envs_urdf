from gym.envs.registration import register
register(
    id='pointRobotUrdf-vel-v0',
    entry_point='urdfenvs.pointRobotUrdf.envs:PointRobotVelEnv'
)
register(
    id='pointRobotUrdf-acc-v0',
    entry_point='urdfenvs.pointRobotUrdf.envs:PointRobotAccEnv'
)
