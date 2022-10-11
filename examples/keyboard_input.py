import gym
from urdfenvs.robots.tiago import TiagoRobot

from multiprocessing import Process, Pipe
import numpy as np
from urdfenvs.keyboard_input.keyboard_input_responder import Responder
from pynput.keyboard import Key


def run_tiago_keyboard(conn, n_steps=10000, render=True):
    robots = [
        TiagoRobot(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    ob = env.reset()
    print(f"Initial observation : {ob}")

    # create zero input action
    action = np.zeros(env.n())
    for i in range(n_steps):

        # request and receive action
        conn.send({"request_action": True, "kill_child": False})
        keyboard_data = conn.recv()

        # update action matrix
        action[0:2] = keyboard_data["action"]
        env.step(action)

    # kill the child properly
    conn.send({"request_action": False, "kill_child": True})


if __name__ == "__main__":

    # setup multi threading with a pipe connection
    parent_conn, child_conn = Pipe()

    # create parent process
    p = Process(target=run_tiago_keyboard, args=(parent_conn,))

    # create Responder object
    responder = Responder(child_conn)

    # unlogical key bindings
    custom_on_press = {
        Key.left: np.array([-1.0, 0.0]),
        Key.space: np.array([1.0, 0.0]),
        Key.page_down: np.array([1.0, 1.0]),
        Key.page_up: np.array([-1.0, -1.0]),
    }

    responder.setup(default_action=np.array([0.0, 0.0]))
    # responder.setup(custom_on_press=custom_on_press)

    # start parent process
    p.start()

    # start child process which keeps responding/looping
    responder.start(p)

    # kill parent process
    p.kill()
