Creating environments
^^^^^^^^^^^^^^^^^^^^^


Examples can be found in the `urdfenvs/examples<https://github.com/maxspahn/gym_envs_urdf/tree/master/examples>`_ folder
you will find individual examples for all implemented
robots. Environments can be created using the normal gym syntax.
Gym environments rely mostly on three functions

- ``gym.make(...)`` to create the environment,
- ``gym.reset(...)`` to reset the environment,
- ``gym.step(action)`` to step one time step in the environment.

For example, in `examples/pointRobot.py
<https://github.com/maxspahn/gym_envs_urdf/blob/master/examples/pointRobot.py>`_, you
can find the following syntax to ``make``, ``reset`` and ``step`` the environment.

.. code:: python

    env = gym.make('pointRobotUrdf-vel-v0', dt=0.05, render=True)
    ob = env.reset(pos=pos0, vel=vel0)
    ob, reward, done, info = env.step(action)

The id-tag in the ``make`` command specifies the robot and the control type.
You can get a full list of all available environments using

.. code:: python

   from gym import envs
   print(envs.registry.all())

Go ahead and explore all the examples you can find there.

Switching
--------

Environments can be created using the normal gym syntax. For example the
below code line creates a toy robot with 3 links.
Actions are velocities to the individual joints.

.. code:: python

    env = gym.make('nLink-urdf-reacher-vel-v0', n=3, dt=0.01, render=True)

A holonomic and a differential drive mobile manipulator are implemented:

.. code:: python

    env = gym.make('albert-reacher-vel-v0', dt=0.01, render=True)
    env = gym.make('mobile-reacher-tor-v0', dt=0.01, render=True)

For most robots, different control interfaces are available, velocity
control, acceleration control and torque control.

