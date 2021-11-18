from tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherVelEnv(TiagoReacherEnv):
    metadata = {"render.modes": ["human"]}

    def applyAction(self, action):
        self.robot.apply_vel_action_wheels(action)
        self.robot.apply_vel_action(action)
