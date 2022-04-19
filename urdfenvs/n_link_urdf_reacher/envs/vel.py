from urdfenvs.n_link_urdf_reacher.envs.n_link_urdf_reacher_env import (
    NLinkUrdfReacherEnv,
)


class NLinkUrdfVelReacherEnv(NLinkUrdfReacherEnv):
    def apply_action(self, action):
        self._robot.apply_velocity_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
