import gym
import tiagoReacher

from pynput import keyboard
from pynput.keyboard import Key, Listener
from multiprocessing import Process, Pipe
import numpy as np
from keyboardInput.keyboard_input_responder import Responder


def main(child_conn):
    env = gym.make('tiago-reacher-vel-v0', dt=0.05, render=True)
    defaultAction = np.zeros(env.n())
    defaultAction[0:2] = np.array([1.0, 0.0])
    defaultAction[10] = 0.0

    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    pos0 = np.zeros(20)
    # base
    pos0[0:3] = np.array([0.0, 1.0, -1.0])
    # torso
    pos0[3] = 0.0
    # head
    pos0[4:6] = np.array([1.0, 0.0])
    # left arm
    pos0[6:13] = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.1, -0.1])
    # right arm
    pos0[13:20] = np.array([-0.5, 0.2, 0.2, 0.0, 0.0, 0.1, -0.1])
    vel0 = np.zeros(19)
    vel0[0:2] = np.array([0.0, 0.0])
    vel0[5:12] = np.array([0.1, 0.1, 0.2, -0.1, 0.1, 0.2, 0.0])

    for e in range(n_episodes):
        ob = env.reset(pos=pos0, vel=vel0)
        print("base: ", ob[0:3])
        print("torso: ", ob[3])
        print("head: ", ob[4:6])
        print("left arm: ", ob[6:13])
        print("right arm: ", ob[13:20])
        print("Starting episode")

        action = defaultAction
        for i in range(n_steps):
            # request and receive action
            # print(' sending action request')
            child_conn.send({"request_action": True})
            keyboard_data = child_conn.recv()

            action = defaultAction
            action[0:2] = keyboard_data["action"]

            print("from main loop: {}".format(i))


            # print(keyboard_data["action"])
            ob, reward, done, info = env.step(action)
            cumReward += reward

    # kill the child properly
    print('kill the child')
    child_conn.send({'kill_child': True})


if __name__ == '__main__':
    # setup multi threading with a pipe connection
    parent_conn, child_conn = Pipe()

    p = Process(target=main, args=(parent_conn,))
    # start child process
    p.start()

    responder = Responder(child_conn)
    # TODO: create more things for setting everything up
    responder.setup()

    # loop to respond to input request
    while p.is_alive():
        # todo: check when child died, if so kill yourself
        # print("request number: {}".format(i))
        # todo: check how stable requests and responds
        responder.respond()




    print("where is my child?")
