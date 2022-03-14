API
=======

Structure
----------

.. inheritance-diagram:: urdfenvs.boxerRobot.resources.boxerRobot.BoxerRobot
        urdfenvs.albertReacher.resources.albertRobot.AlbertRobot
        urdfenvs.tiagoReacher.resources.tiagoRobot.TiagoRobot
        urdfenvs.pandaReacher.resources.pandaRobot.PandaRobot
        urdfenvs.mobileReacher.resources.mobileRobot.MobileRobot
        urdfenvs.nLinkUrdfReacher.resources.nLinkRobot.NLinkRobot
        urdfenvs.pointRobotUrdf.resources.pointRobot.PointRobot
      :top-classes: urdfenvs.urdfCommon.generic_robot.GenericRobot
          urdfenvs.urdfCommon.holonomic_robot.HolonomicRobot
          urdfenvs.urdfCommon.differential_drive_robot.DifferentialDriveRobot
      :parts: 1

Generic Classes
---------------

UrdfEnv
^^^^^^^

.. autoclass:: urdfenvs.urdfCommon.urdfEnv.UrdfEnv
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

.. autoclass:: urdfenvs.urdfCommon.urdfEnv.WrongObservationError
   :inherited-members: 
   :show-inheritance:

