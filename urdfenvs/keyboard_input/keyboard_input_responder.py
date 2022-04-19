from pynput import keyboard
from pynput.keyboard import Key
import numpy as np
import warnings

# reserved keys are detected, but not executed by the gym environment
reserved_keys = [
    Key.esc,
    "q",
    "w",
    "e",
    "r",
    "t",
    "y",
    "u",
    "i",
    "o",
    "p",
    "a",
    "s",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "z",
    "x",
    "c",
    "v",
    "b",
    "n",
    "m",
    "<",
    ",",
    ">",
    ".",
    "/",
    "?",
    ":",
    ";",
    "'",
    "[",
    "]",
    "{",
    "}",
    '"',
    "|",
    "`",
    "~",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "_",
    "-",
    "=",
    "+",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
]


class Responder:
    """
    The Responder class listens to keyboard input from the user and updates the
    class variable action.  An action can be pulled with the respond function.
    Using this pattern has 2 advantages:
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
        self._conn = child_conn
        self._default_action = None
        self._action = None

    def _on_press_outer(self, custom_on_press=None):
        """
        Updating the class variable action on key press
        :param custom_on_press: Dictionary with key action bindings
        """
        if custom_on_press is None:
            # default action keybindings
            def on_press(key):
                if key == Key.up:
                    self.action = np.array([1.0, 0.0])
                if key == Key.down:
                    self.action = np.array([-1.0, 0.0])
                if key == Key.left:
                    self.action = np.array([0.0, 1.0])
                if key == Key.right:
                    self.action = np.array([0.0, -1.0])

        else:
            # custom action keybindings
            def on_press(key):
                # print("key pressed: {}".format(key))
                for custom_key in custom_on_press:
                    if key == custom_key:
                        self.action = custom_on_press[custom_key]

        return on_press

    def _on_release_outer(self, custom_on_release=None):
        """
        Updating the class variable action to default action on key release
        :param custom_on_release: Dictionary with key action bindings
        """
        if custom_on_release is None:
            # default action keybindings
            def on_press(key):
                self.action = self.default_action

        else:
            # custom action keybindings
            def on_press(key):
                for custom_key in custom_on_release:
                    if key == custom_key:
                        self.action = custom_on_release[custom_key]

        return on_press

    def respond(self):
        """Respond to request with the latest action"""

        # receive request
        request = self.conn.recv()
        if request["request_action"]:
            # send action
            self.conn.send({"action": self._action})
        elif request["kill_child"]:
            raise Exception
        else:
            raise Exception("Cannot handle action request")

    def setup(
        self,
        default_action=np.array([0.0, 0.0]),
        custom_on_press=None,
        custom_on_release=None,
    ):
        """
        Setup responder, optionally set defaultaction and custom action
        keybindings

        Parameters
        ----------

        default_action: np.ndarray
            Default action when no action is specified
        custom_on_press:
            Custom on press callback function for key bindings
        custom_on_release:
            Custom on release callback function for key bindings
        """
        self.default_action = default_action
        self.action = default_action

        # warn if a reserved key is in the custom_on_press keys
        if custom_on_press is not None:
            if (
                len(
                    [
                        key
                        for key in reserved_keys
                        if key in custom_on_press.keys()
                    ]
                )
                > 0
            ):
                warnings.warn("reserved key used for control")

        # warn if a reserved key is in the custom_on_release keys
        if custom_on_release is not None:
            if (
                len(
                    [
                        key
                        for key in reserved_keys
                        if key in custom_on_release.keys()
                    ]
                )
                > 0
            ):
                warnings.warn("reserved key used for control")

        # setup keyboard listener
        listener = keyboard.Listener(
            on_press=self._on_press_outer(custom_on_press),
            on_release=self._on_release_outer(custom_on_release),
        )
        # start listening to keyboard input
        listener.start()

    def start(self, p):
        """
        start with responding to keyboard input

        :param p: Parent process
        """

        # while parent process is alive, keep on responding
        while p.is_alive():
            try:
                self.respond()
            except Exception:
                return

    # getters and setters
    @property
    def conn(self):
        return self._conn

    @property
    def default_action(self):
        return self._default_action

    @default_action.setter
    def default_action(self, value):
        self._default_action = value

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, value):
        self._action = value
