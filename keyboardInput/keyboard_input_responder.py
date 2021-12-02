from pynput import keyboard
from pynput.keyboard import Key
import numpy as np


class Responder:
    """
    The Responder class listens to keyboard input from the user and updates the class variable action.
    An action can be pulled with the respond function. Using this pattern has 2 advantages:
        * keyboard listener frequency does not depend on game loop frequency
        * actions are only send if they are going to be processed

    Usage
    initializing and set up:
        * Pipe between parent and child processes
        * Main loop(parent_connection) containing the main loop
        * Responder(child_connection) object

    In the main loop ask for an action and wait for the responder to
    respond as follows:
        parent_connection.send({"request_action": True})
        keyboard_data = parent_connection.recv()
    """

    def __init__(self, child_conn):
        """
        :param child_conn: Pipe connection to the parent process
        """

        self.__conn = child_conn
        self.__defaultAction = None
        self.__action = None

    def __on_press_outer(self, custom_on_press=None):
        """ Updating the class variable action on key press"""

        if custom_on_press is None:
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
        else:
            # add custom action keybindings
            def on_press(key):
                for custom_key in custom_on_press:
                    if key == custom_key:
                        self.__action = custom_on_press[custom_key]
        return on_press

    def __on_release_outer(self, custom_on_release=None):
        """ Updating the class variable action to default action on key release"""

        if custom_on_release is None:
            # default action keybindings
            def on_press(key):
                # print("key released: {}".format(key))
                self.__action = self.__defaultAction
        else:
            # custom action keybindings
            def on_press(key):
                for custom_key in custom_on_release:
                    if key == custom_key:
                        self.__action = custom_on_release[custom_key]
        return on_press

    def respond(self):
        """ Respond to request with the latest action """

        # receive request
        request = self.__conn.recv()
        if request["request_action"]:
            # send action
            self.__conn.send({"action": self.__action})
        elif request["kill_child"]:
            raise Exception
        else:
            # TODO: could request different kinds of actions
            raise Exception("Cannot handle action request")

    def setup(self, defaultAction=np.array([0.0, 0.0]), custom_on_press=None, custom_on_release=None):
        """
        Setup responder, optionally set defaultaction and custom action keybindings

        :param defaultAction:   Default action when no action is specified
        :param custom_on_press:     custom on press callback function for key bindings
        :param custom_on_release:   custom on release callback function for key bindings
        :return:
        """
        self.__defaultAction = defaultAction
        self.__action = defaultAction

        # setup keyboard listener
        listener = keyboard.Listener(
            on_press=self.__on_press_outer(custom_on_press),
            on_release=self.__on_release_outer(custom_on_release)
        )
        # start listening to keyboard input
        listener.start()

    def start(self, p):
        """
        start with responding to keyboard input

        :param p: Parent process
        :return:
        """

        # while parent process is alive
        while p.is_alive():
            try:
                self.respond()
            except Exception:
                return

    # TODO: proper way of getters and setter for private class variables
    # @property
    # def action(self):
    #     return self.__action
    #
    # @action.setter
    # def action(self, value):
    #     self.__action = value
