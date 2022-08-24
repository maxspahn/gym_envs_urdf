import gym
import urdfenvs.boxer_robot # pylint: disable=unused-import
import numpy as np


def main():
    env = gym.make("boxer-robot-vel-v0", dt=0.01, render=True)
    default_action = np.array([0.6, 0.8])
    n_episodes = 1
    n_steps = 100000
    cum_reward = 0.0
    pos0 = np.array([1.0, 0.2, -1.0]) * 0.0
    for _ in range(n_episodes):
        ob = env.reset(pos=pos0)
        print(f"Initial observation : {ob}")
        env.add_walls()
        print("Starting episode")
        for _ in range(n_steps):
            action =default_action
            ob,  reward, _ = env.step(action)
            cum_reward += reward


if __name__ == "__main__":
    main()
