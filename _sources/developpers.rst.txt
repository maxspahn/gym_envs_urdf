Robot Structure
================

The main purpose of this repository is the fast implementation of robots into an
OpenAI-Gym environment. This relies on the urdf files and information about which joints
result in motion. In the following, some details on the structure are elaborated.

Generic Robot
---------------

The python-class ``GenericRobot`` in ``urdfCommon/generic_robot.py`` is an abstract class that
every robot inherits from. Abstract methods are used to force some methods to be
implemented that are required by the environment. There are two intermediate classes,
namely ``DifferentialDriveRobot`` and ``HolonomicRobot`` that derive from ``GenericRobot``. 
The main difference between them is ``update_state`` function as the ``differential_drive_robot``
requires additional information.

Robot States
----------------

All environments follow the OpenAI-Gym convention of returning the observation (reward,
done, info). The observation is a dictionary containing as keys ``x``, ``xdot`` and optional
observations. ``x`` and ``xdot`` are numpy arrays for the position and the velocity in each
degree of freedom. For example, the panda robot returns two 7-dimensional arrays
containing the joint positions and joint velocities. For non-holonomic robots, such as the
boxer, the observation also contains information about forward and rotational velocity
under the key ``vel``. Note that this information is redundant with ``xdot`` but we concluded
that it might be useful for some methods and provide this information explicitly.

