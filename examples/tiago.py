import gym
from urdfenvs.robots.tiago import TiagoRobot
import numpy as np

def run_tiago(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        TiagoRobot(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(env.n())
    action[0:2] = np.array([0.2, 0.02])
    action[10] = 0.0
    pos0 = np.zeros(20)
    pos0[0] = -1.7597e-1
    pos0[3] = 0.1
    vel0 = np.zeros(19)
    ob = env.reset(pos=pos0, vel=vel0)
    print("base: ", ob['robot_0']["joint_state"]["position"][0:3])
    print("torso: ", ob['robot_0']["joint_state"]["position"][3])
    print("head: ", ob['robot_0']["joint_state"]["position"][4:6])
    print("left arm: ", ob['robot_0']["joint_state"]["position"][6:13])
    print("right arm: ", ob['robot_0']["joint_state"]["position"][13:20])
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_tiago(render=True)
