import gym
import numpy as np

"""
Unfortunately, Stable Baselines3 has a problem with `float64` action spaces in DDPG, TD3 and SAC algorithms. This wrapper solves that problem by converting the `dType` of action space to `float32`.
"""
class StableBaselinesFloat32ActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=self.env.action_space.low.astype(np.float32),
            high=self.env.action_space.high.astype(np.float32),
            dtype=np.float32,
        )

    def step(self, action):
        action = action.astype(np.float32)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info