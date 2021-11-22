from tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherVelEnv(TiagoReacherEnv):
    metadata = {"render.modes": ["human"]}

    def applyAction(self, action):
        self.robot.apply_base_velocity(action)
        self.robot.apply_vel_action(action)
