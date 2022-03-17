import gym
import urdfenvs.prius
import numpy as np


def main():
    env = gym.make("prius-vel-v0", dt=0.01, render=True)
    action = np.array([1.1, 0.0])
    n_steps = 1000
    pos0 = np.array([-1.0, 0.2, -1.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if ob['steering'] > 0.2:
            action[0] = 0
        #print(ob['steering'])
        print(ob['vel'][0])



if __name__ == "__main__":
    main()
