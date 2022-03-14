from gym.envs.registration import register
register(
    id='pointRobotUrdf-vel-v0',
    entry_point='urdfenvs.point_robot_urdf.envs:PointRobotVelEnv'
)
register(
    id='pointRobotUrdf-acc-v0',
    entry_point='urdfenvs.point_robot_urdf.envs:PointRobotAccEnv'
)
