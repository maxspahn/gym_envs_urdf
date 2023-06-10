# Basic imports
import gym
import numpy as np
import os
from gym.wrappers import FlattenObservation

# Stable baselines 3
import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3 import MlpPolicy

print(stable_baselines3.__version__)

# URDF Envs
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.urdf_common.reward import Reward
from urdfenvs.wrappers.stable_baselines_float32_action_wrapper import StableBaselinesFloat32ActionWrapper


MODEL_NAME = 'TD3-001'


models_dir = 'models/' + MODEL_NAME
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


class InverseDistanceDenseReward(Reward):
    def calculateReward(self, observation: dict) -> float:
        goal = observation['robot_0']['FullSensor']['goals'][1]['position']
        position = observation['robot_0']['joint_state']['position']
        reward = 1.0/np.linalg.norm(goal-position)
        # print(f'🏆 Reward is: {reward}')
        return reward


robots = [
    GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
]

env = UrdfEnv(
    dt=0.01,
    robots=robots,
    render=False,
)


env.add_goal(goal1)
sensor = FullSensor(['position'], ['position', 'size'], variance=0.0)
env.add_sensor(sensor, [0])
env.set_spaces()
env.set_reward_calculator(InverseDistanceDenseReward())
defaultAction = np.array([0.5, -0.0, 0.0])
pos0 = np.array([0.0, 0.1, 0.0])
vel0 = np.array([1.0, 0.0, 0.0])
ob = env.reset(pos=pos0, vel=vel0)
env.shuffle_goals()

env = FlattenObservation(env)

env = StableBaselinesFloat32ActionWrapper(env)

print(f"🚀 Action Space Data Type: {env.action_space.sample().dtype}, Observation Space Data Type: {env.observation_space.sample().dtype}")
print(type(env.action_space))

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

TIMESTEPS = 100000
model = TD3(
    MlpPolicy, 
    env,
    action_noise=action_noise,
    verbose=0, 
    tensorboard_log=logdir,
    policy_kwargs={"net_arch": [64, 64]},
        )
model.learn(total_timesteps=TIMESTEPS, 
            log_interval=10,
            tb_log_name=MODEL_NAME, 
            progress_bar=False, 
        )
# model.save(f"{MODEL_NAME}-model")
print("🏁 Training complete!")