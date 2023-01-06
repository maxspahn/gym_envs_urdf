import pybullet
from urdfenvs.urdf_common.physics_engine import PhysicsEngine


class BulletPhysicsEngine(PhysicsEngine):
    def __init__(self, render: bool):
        if render:
            self._cid = pybullet.connect(pybullet.GUI)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self._cid = pybullet.connect(pybullet.DIRECT)

    def configure(self, dt: float, num_sub_steps: int):
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=dt, numSubSteps=num_sub_steps
        )
        pybullet.setGravity(0, 0, -10.0)

    def id(self):
        return self._cid

    def step(self):
        pybullet.stepSimulation(self._cid)

    def add_goal(self, goal):
        return goal.add_to_bullet(pybullet)

    def add_obstacle(self, obstacle):
        return obstacle.add_to_bullet(pybullet)

    def update_goal(self, goal, t: float):
        goal.update_bullet_position(pybullet, t=t)

    def update_obstacle(self, obstacle, t: float):
        obstacle.update_bullet_position(pybullet, t=t)

    def close(self):
        pybullet.disconnect(self._cid)

    def extract_relevant_joint_ids(self, joint_names, robot):
        robot_joints = []
        castor_joints = []
        num_joints = pybullet.getNumJoints(robot)
        for name in joint_names:
            for i in range(num_joints):
                joint_info = pybullet.getJointInfo(robot, i)
                joint_name = joint_info[1].decode("UTF-8")
                if joint_name == name:
                    robot_joints.append(i)
            for i in range(num_joints):
                joint_info = pybullet.getJointInfo(robot, i)
                joint_name = joint_info[1].decode("UTF-8")
                if "castor" in joint_name:
                    castor_joints.append(i)
        return robot_joints, castor_joints

    def disable_velocity_control(self, robot, robot_joints):
        """Disables velocity control for all controlled joints.

        By default, pybullet uses velocity control. This has to be disabled if
        torques should be directly controlled.  See
        func:`~urdfenvs.urdfCommon.generic_robot.generic_rob
        ot.apply_torque_action`
        """
        friction = 0.0
        n = len(robot_joints)
        for i in range(n):
            pybullet.setJointMotorControl2(
                robot,
                jointIndex=robot_joints[i],
                controlMode=pybullet.VELOCITY_CONTROL,
                force=friction,
            )
