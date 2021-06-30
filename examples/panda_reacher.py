import gym
import pandaReacher
import time
import numpy as np


def main():
    #env = gym.make('panda-reacher-tor-v0', dt=0.01, render=True)
    gripper = True
    env = gym.make('panda-reacher-acc-v0', dt=0.01, render=True, gripper=gripper)
    #env = gym.make('panda-reacher-vel-v0', dt=0.01, render=True)
    defaultAction = np.ones(9) * 0.0
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset()
        print("Starting episode")
        for i in range(n_steps):
            if (int(i/100))%2 == 0:
                defaultAction[7] = -1.0
                defaultAction[8] = -1.0
            else:
                defaultAction[7] = 1.0
                defaultAction[8] = 1.0
            time.sleep(env._dt)
            action = env.action_space.sample()
            if gripper:
                action = defaultAction[0:9]
            else:
                action = defaultAction[0:7]
            ob, reward, done, info = env.step(action)
            #print(ob[8])
            cumReward += reward


if __name__ == '__main__':
    main()
