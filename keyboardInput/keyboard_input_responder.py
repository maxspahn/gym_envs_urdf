from pynput import keyboard
from pynput.keyboard import Key, Listener
import numpy as np


class Responder:
    """
    The Responder class listens to keyboard input from the user and updates the class variable action.
    An action can be pulled with the respond function. Using this pattern has 2 advantages:
        * keyboard listener frequency does not depend on game loop frequency
        * actions are only send if they are going to be processed

    After initializing and setting up the Responder object, request an action and wait for
    action in the main game loop.

    Example:
            for i in range(n_steps):
                child_conn.send({"request_action": True})   # request action
                keyboard_data = child_conn.recv()           # wait for requested action
                action = keyboard_data["action"]
                env.step(action)
    """

    def __init__(self, child_conn):
        self.__conn = child_conn
        self.__defaultAction = None
        self.__action = None

    def __on_press_outer(self):
        """ Updating the class variable action on key press"""

        def on_press(key):
            # print("key pressed: {}".format(key))

            # default action keybindings
            if key == 'w' or key == Key.up:
                self.__action = np.array([1.0, 0.0])
            if key == 's' or key == Key.down:
                self.__action = np.array([-1.0, 0.0])
            if key == 'a' or key == Key.left:
                self.__action = np.array([0.0, 1.0])
            if key == 'd' or key == Key.right:
                self.__action = np.array([0.0, -1.0])

        return on_press

    def __on_release_outer(self):
        """ Updating the class variable action to default action on key release"""

        def on_press(key):
            # print("key released: {}".format(key))

            self.__action = self.__defaultAction

        return on_press

    def setup(self, defaultAction=np.array([0.0, 0.0])):
        """ Setup responder, optionally set defaultaction, custom action keybindings """
        self.__defaultAction = defaultAction
        self.__action = defaultAction

        # setup keyboard listener
        listener = keyboard.Listener(
            # TODO: option to create self made on press functions
            on_press=self.__on_press_outer(),
            on_release=self.__on_release_outer()
        )
        # start listening to keyboard input
        listener.start()

    def respond(self):
        """ Respond to request with the latest action """

        # receive request
        request = self.__conn.recv()
        if request["request_action"]:
            # send action
            self.__conn.send({"action": self.__action})
        else:
            # TODO: could request different kinds of actions
            raise Exception("Cannot handle action request")
