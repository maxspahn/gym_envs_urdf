Dependencies
============

Scenes
-------
todo...


Robot control with the keyboard
-------------------------------

Control robot actuators with keyboard keys. This is done by:

- Setting up a parent en child process with a pipe connection inbetween
- Setup and start main process with parent\_connection as argument
- Setup Responder object with child\_connection as argument
- Start Responder with parent process as argument

In the main loop an request for action should be made followed by
waiting for a response as such:

.. code:: python

    parent_conn.send({"request_action": True})
    keyboard_data = parent_conn.recv()
    action = keyboard_data["action"]

Additionally custom key bindings and a default action can and passed as arguement
to the responder. An example can be found in `urdfenvs/examples/keyboard_input.py
<https://github.com/maxspahn/gym_envs_urdf/blob/master/examples/keyboard_input.py>`_.
