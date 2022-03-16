API
=======

Structure
----------

.. inheritance-diagram:: urdfenvs.boxer_robot.resources.boxer_robot.BoxerRobot
        urdfenvs.albert_reacher.resources.albert_robot.AlbertRobot
        urdfenvs.tiago_reacher.resources.tiago_robot.TiagoRobot
        urdfenvs.panda_reacher.resources.panda_robot.PandaRobot
        urdfenvs.mobile_reacher.resources.mobile_robot.MobileRobot
        urdfenvs.n_link_urdf_reacher.resources.n_link_robot.NLinkRobot
        urdfenvs.point_robot_urdf.resources.point_robot.PointRobot
      :top-classes: urdfenvs.urdfCommon.generic_robot.GenericRobot
          urdfenvs.urdfCommon.holonomic_robot.HolonomicRobot
          urdfenvs.urdfCommon.differential_drive_robot.DifferentialDriveRobot
      :parts: 1

Generic Classes
---------------

UrdfEnv
^^^^^^^

.. autoclass:: urdfenvs.urdfCommon.urdf_env.UrdfEnv
   :inherited-members: 
   :show-inheritance:

GenericRobot
^^^^^^^^^^^^^

.. autoclass:: urdfenvs.urdfCommon.generic_robot.GenericRobot
   :inherited-members: 
   :show-inheritance:

DifferentialDriveRobot
***********************

.. autoclass:: urdfenvs.urdfCommon.differential_drive_robot.DifferentialDriveRobot
   :inherited-members: 
   :show-inheritance:

HolonomicRobot
**************

.. autoclass:: urdfenvs.urdfCommon.holonomic_robot.HolonomicRobot
   :inherited-members: 
   :show-inheritance:

Exceptions
--------------

.. autoclass:: urdfenvs.urdfCommon.urdf_env.WrongObservationError
   :inherited-members: 
   :show-inheritance:

