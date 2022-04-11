from gym.envs.registration import register
register(
    id='dual-arm-vel-v0',
    entry_point='urdfenvs.dual_arm.envs:DualArmVelEnv'
)
register(
    id='dual-arm-acc-v0',
    entry_point='urdfenvs.dual_arm.envs:DualArmAccEnv'
)
