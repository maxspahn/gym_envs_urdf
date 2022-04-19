import gym
import urdfenvs.boxer_robot
import numpy as np


def main():
    env = gym.make("boxer-robot-vel-v0", dt=0.01, render=True)
    defaultAction = np.array([0.6, 0.8])
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    pos0 = np.array([1.0, 0.2, -1.0]) * 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=pos0)
        print(f"Initial observation : {ob}")
        env.set_walls(limits=[[-4, -4], [4, 4]])
        print("Starting episode")
        for i in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == "__main__":
    main()
