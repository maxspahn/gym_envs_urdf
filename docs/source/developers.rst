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

All environments follow the OpenAI-Gym convention of returning observation, (reward,
done, info). The observation is a dictionary containing as keys ``joint_state`` and optional
observations from added sensors.

HolonomicRobot
^^^^^^^^^^^^^^

``joint_state`` is a dictionary containing nested dictionaries:

    ``position``: a numpy array containing the joint positions in for each degree of freedom.

    ``velocity``: a numpy array containing the joint velocities in for each degree of freedom.

As example, the Panda robot has 7 joints. The numpy array ``ob["joint_state"]["position"]`` and
``ob["joint_state"]["velocity"]`` both have dimension (7, ).

DifferentialDriveRobot
^^^^^^^^^^^^^^^^^^^^^^

``joint_state`` is a dictionary containing nested dictionaries:

    ``position``: a numpy array containing the the joint positions for each degree of freedom.
    the base joint's configuration space equals the world frame and has 3 dimensions.
    x, y position and the orientation of the base joints' center of mass. Which is concatenated with
    the other joint positions.

    ``velocity``: a numpy array containing all joint velocities. Just as the positions the base is
    3-dimensional.

    ``forward_velocity``: a float with the forward velocity in robot frame.

Note that ``forward_velocity`` is redundant with ``velocity`` but we concluded
that it might be useful for some methods and provide this information explicitly.

As example, the Albert robot (a mobile base with a Panda arm attached) has 8 joints.
array ``ob["joint_state"]["position"]`` and
``ob["joint_state"]["velocity"]`` both have dimension (10, ). For positions (and similar for velocities)
in format ``(base_x_pos, base_y_pos, base_orientation, joint_2_pos, joint_3_pos, ..., joint_8_pos)``


