import numpy as np
import gymnasium as gym
from urdfenvs.urdf_common.generic_mujoco_env import GenericMujocoEnv

env: GenericMujocoEnv = gym.make("generic-mujoco-env-v0", xml_file='panda_scene.xml', render_mode='human')


observation, info = env.reset(seed=42)
pos0=np.array([0,0,0,-1.57079,0,1.57079,-0.7853,0.04,0.04])
env.reset_model(pos=pos0)

action_mag = np.array([0.3, -0.2, 0.3, -0.15, 0.2, -0.01, 0.35, 0.01])
N = 300
t = 0.0
for _ in range(N):
    t += env.dt
    action = action_mag * np.cos(t)
    observation, *_ = env.step(action)
env.close()
