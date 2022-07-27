Adding elements to the environment
===================================

Walls
-----

To add a wall use:

.. code:: python

    env.add_walls(dim, poses_2d)


dim: numpy.ndarray or list
| (optional) dim, short for dimensions if given dim must
| be of length 3 corresponding to [width, length, height].

| poses_2d: [np.ndarray, ...] or [list, ...]
| (optional) if given poses_2d must be a list with lists or numpy arrays of length 3
| correspond to [x_position, y_position, orientation] which refers to walls centers of mass.
| The height of the center of mass will be adjusted such that the bottom of the wall
| aligns with the ground plane. multiple walls can be placed
| by having multiple pose_2d in poses_2d.

Shapes
-------

To add a shape use:

.. code:: python

    env.add_shapes(shape_type, dim, mass, poses_2d, place_height)


| shape_type: str
| (required) the 4 options available are:
| "GEOM_SPHERE", "GEOM_BOX", "GEOM_CYLINDER", "GEOM_CAPSULE".

| dim: numpy array of list
| (optional), if dimensions is specified the length dependents on the shape_type:
|     GEOM_SPHERE,    dim=[radius],
|     GEOM_BOX,       dim=[width, length, height],
|     GEOM_CYLINDER,  dim=[radius, length],
|     GEOM_CAPSULE,   dim=[radius, length],

| mass: int or float
| (optional) mass refers to the objects mass, which in uniformly distributed.

| poses_2d: [np.ndarray, ...] or [list, ...]
| (optional) if specified poses_2d must be a list containing lists or numpy arrays of length 3
| correspond to [x_position, y_position, orientation] indicates the shapes center of mass.
| multiple walls can be placed by having multiple pose_2d in poses_2d.

| place_height: int or float
| (optional) if specified it refers to the z_position of the center of mass
| if not specified, the shape will be placed such that the shapes bottom
| aligns with the ground plane.

URDF files
---------------
todo


Sensors
--------

A robot can be given a lidar or obstacle sensor by creating a sensor
object and passing it to the environment:

.. code:: python

    sensor = ObstacleSensor()
    env.add_sensor(sensor)

The observations from the sensor are returned by the ``env.step(action)`` call.
The structure of the observation varies depending on the sensor and its arguments
for more info see the Sensor class and subclasses located at urdfenvs/sensors/


Goals
------
todo..
