Adding obstacles to the environment
===================================

Walls
-----



Shapes
-------
todo

URDF Obstacles
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
