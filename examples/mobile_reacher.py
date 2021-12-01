import gym
import mobileReacher
import numpy as np


def main():
    #env = gym.make('mobile-reacher-tor-v0', dt=0.01, render=True)
    #env = gym.make('mobile-reacher-vel-v0', dt=0.01, render=True)
    env = gym.make('mobile-reacher-acc-v0', dt=0.01, render=True, gripper=False)
    defaultAction = np.zeros(10)
    defaultAction[0] = 0.1
    defaultAction[5] = -0.0
    defaultAction[-1] = 3.5
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=np.zeros(10), vel=np.zeros(10))
        print("Starting episode")
        for i in range(n_steps):
            if (int(i/100))%2 == 0:
                defaultAction[-1] = -1.0
            else:
                defaultAction[-1] = 1.0
            action = env.action_space.sample()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    main()
