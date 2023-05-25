# Basic imports
import gym
import numpy as np
import os
from gym.wrappers import FlattenObservation

# Stable baselines 3
''' 
Stable baselines 3 has a build-in function called `check_env` that checks if the environment is compatible with the library. 
It checks the following:
    - Observation space
    - Action space
    - Reward range
    - Whether the environment is vectorized or not
    - Whether the environment uses a `Dict` or `Tuple` observation space
    - Whether the environment uses a `Dict` or `Tuple` action space
    - Whether the environment uses a `VecEnv` or not
    - Whether the environment uses a `VecNormalize` wrapper or not
    - Whether the environment uses a `FlattenObservation` wrapper or not
    - Whether the environment uses a `FrameStack` wrapper or not
    - Whether the environment uses a `TimeLimit` wrapper or not
    - Whether the environment uses a `Monitor` wrapper or not
    - Whether the environment uses a `VecFrameStack` wrapper or not
    - Whether the environment uses a `VecTransposeImage` wrapper or not
'''
from stable_baselines3.common.env_checker import check_env


# URDF Envs
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.urdf_common.reward import Reward


class InverseDistanceDenseReward(Reward):
    def calculateReward(self, observation: dict) -> float:
        goal = observation['robot_0']['FullSensor']['goals'][1]['position']
        position = observation['robot_0']['joint_state']['position']
        reward = 1.0/np.linalg.norm(goal-position)
        print(f'🏆 Reward is: {reward}')
        return reward
    



robots = [
    GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
env= UrdfEnv(
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

check_env(env, warn=True)