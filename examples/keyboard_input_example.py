import gym
import tiagoReacher

from multiprocessing import Process, Pipe
import numpy as np
from keyboardInput.keyboard_input_responder import Responder
from pynput.keyboard import Key


def main(conn):
    # copy of examples/tiago.py
    env = gym.make("tiago-reacher-vel-v0", dt=0.01, render=True)
    n_steps = 1000
    pos0 = np.zeros(20)
    vel0 = np.zeros(19)
    ob = env.reset(pos=pos0, vel=vel0)

    action = np.zeros(env.n())
    for i in range(n_steps):
        # request and receive action
        conn.send({"request_action": True, "kill_child": False})
        keyboard_data = conn.recv()

        action[0:2] = keyboard_data["action"]

        ob, reward, done, info = env.step(action)

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

    responder.setup(defaultAction=np.array([1.0, 0.0]))
    # responder.setup(custom_on_press=custom_on_press)

    # start child process which keeps responding/looping
    responder.start(p)

    # kill parent process
    p.kill()
