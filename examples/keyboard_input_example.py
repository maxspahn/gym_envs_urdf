import gym
import tiagoReacher

from multiprocessing import Process, Pipe
import numpy as np
from keyboardInput.keyboard_input_responder import Responder
from pynput.keyboard import Key


def main(conn):
    # copy of examples/tiago.py
    env = gym.make("tiago-reacher-vel-v0", dt=0.01, render=True)
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
            conn.send({"request_action": True, "kill_child": False})
            keyboard_data = conn.recv()

            action[0:2] = keyboard_data["action"]

            ob, reward, done, info = env.step(action)
            cumReward += reward

    # kill the child properly
    conn.send({"request_action": False,
               "kill_child": True})

if __name__ == "__main__":
    # setup multi threading with a pipe connection
    parent_conn, child_conn = Pipe()

    # create and start parent process
    p = Process(target=main, args=(parent_conn,))
    p.start()

    # create Responder object
    responder = Responder(child_conn)

    # unlogical key bindings
    custom_on_press = {Key.up: np.array([-1.0, 0.0]),
                       Key.down: np.array([1.0, 0.0]),
                       Key.left: np.array([1.0, 1.0]),
                       Key.right: np.array([-1.0, -1.0])}

    responder.setup()
    # responder.setup(custom_on_press=custom_on_press)

    # start child process which keeps responding/looping
    responder.start(p)

    # kill parent process
    p.kill()
