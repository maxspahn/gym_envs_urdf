Getting started
===================

This is the guide to quickle get going with urdf gym environments.

Pre-requisites
----------------

- Python >3.6, <3.10
- pip3
- git 


Installation
------------

You first have to downlad the repository

.. code:: bash

   git clone git@github.com:maxspahn/gym_envs_urdf.git

Then, you can install the package using pip as:

.. code:: bash
   
   pip3 install .

Optional: Installation with poetry
------------------------------------

If you want to use `poetry <https://python-poetry.org/docs/>`_, you have to install it
first. See their webpage for instructions `docs <https://python-poetry.org/docs/>`_. Once
poetry is installed, you can install the virtual environment with the following commands.
Note that during the first installation ``poetry update`` takes up to 300 secs.

.. code:: bash

    poetry install

The virtual environment is entered by

.. code:: bash

    poetry shell

Inside the virtual environment you can access all the examples.

Examples
-----------

Run example
^^^^^^^^^^^

You find several python scripts in `examples/
<https://github.com/maxspahn/gym_envs_urdf/tree/master/examples>`_. You can
test those examples using the following (if you use poetry, make sure to enter the virtual
environment first with ``poetry shell``)

.. code:: python

   python3 pointRobot.py

Replace pointRobot.py with the name of the script you want to run.

Use environments
^^^^^^^^^^^^^^^^


In the ``examples``, you will find individual examples for all implemented 
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


Go ahead and explore all the examples you can finde there.

