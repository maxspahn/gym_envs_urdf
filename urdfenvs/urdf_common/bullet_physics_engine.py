import pybullet
import numpy as np
from urdfenvs.urdf_common.plane import Plane
from urdfenvs.urdf_common.physics_engine import PhysicsEngine


class BulletPhysicsEngine(PhysicsEngine):
    def __init__(self, render: bool):
        super().__init__(render)
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
        self.plane = Plane()

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

    def disable_lateral_friction(self, robot, robot_joints):
        for i in robot_joints:
            pybullet.changeDynamics(robot, i, lateralFriction=0)

    def reset_simulation(self):
        pybullet.resetSimulation()

    def load_urdf(
        self,
        urdf_file: str,
        mount_position: np.ndarray,
        mount_orientation: np.ndarray,
        scaling: float = 1.0,
    ):

        return pybullet.loadURDF(
            fileName=urdf_file,
            basePosition=mount_position.tolist(),
            baseOrientation=mount_orientation.tolist(),
            flags=pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            globalScaling=scaling,
        )

    def set_initial_joint_states(
            self,
            robot,
            robot_joints: list,
            initial_positions: np.ndarray,
            initial_velocities: np.ndarray
    ):
        for i in range(initial_positions.size):
            pybullet.resetJointState(
                robot,
                robot_joints[i],
                initial_positions[i],
                targetVelocity=initial_velocities[i],
            )

    def apply_torque_action(self, torques: np.ndarray, robot, robot_joints: list) -> None:
        for i in range(torques.size):
            pybullet.setJointMotorControl2(
                robot,
                robot_joints[i],
                controlMode=pybullet.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_velocity_action(self, vels: np.ndarray, robot, robot_joints: list) -> None:
        for i in range(vels.size):
            pybullet.setJointMotorControl2(
                robot,
                robot_joints[i],
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def apply_position_action(self, poss: np.ndarray, robot, robot_joints: list) -> None:
        for i in range(poss.size):
            pybullet.setJointMotorControl2(
                robot,
                robot_joints[i],
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=poss[i],
            )

    def joint_states(self, robot, robot_joints: list):
        joint_pos_list = []
        joint_vel_list = []
        for i in range(len(robot_joints)):
            pos, vel, _, _ = pybullet.getJointState(robot, robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)
        return joint_pos, joint_vel


    def get_base_state(self, robot, robot_joints: list, correct_base_orientation) -> tuple:
        """Updates the robot state.

        The robot state is stored in the self.state, which contains
        a dictionary with key 'joint_state' with nested dictionaries:
        `position`: np.array((base_pose2D, joint_position_2, ...,
            joint_position_n))
            the position in local configuration space
            the base's configuration space aligns with the world frame
            base_pose2D = (x_positions, y_position, orientation)
            the joints 2 to n have al 1-dimensional configuration space
            joint_position_i = (position in local configuration space)
        `velocity`: np.array((base_twist2D, joint_velocity_2, ...,
            joint_velocity_n))
            the velocity in local configuration space
            the base's configuration space aligns with the world frame
            base_pose2D = (x_positions, y_position, orientation)
            the joints 2 to n have al one dimensional configuration space
            joint_velocity_i = (position in local configuration space)
        `forward_velocity`: float
            forward velocity in robot frame
        """
        # base position
        link_state = pybullet.getLinkState(robot, 0, computeLinkVelocity=0)
        pos_base = np.array(
            [
                link_state[0][0],
                link_state[0][1],
                pybullet.getEulerFromQuaternion(link_state[1])[2],
            ]
        )

        # make sure that the rotation is within -pi and pi
        base_position = correct_base_orientation(pos_base)

        # wheel velocities
        vel_wheels = pybullet.getJointStates(robot, robot_joints)
        wheel_velocity = [vel_wheels[0][1], vel_wheels[1][1]]
        return base_position, wheel_velocity

    def get_link_state(self, robot, link_id):
        return pybullet.getLinkState(robot, link_id, computeLinkVelocity=1)


    def apply_thrust(self, robot, robot_joints, thrusts: np.ndarray):
        for i in range(len(robot_joints)):
            pybullet.applyExternalForce(robot,
                                 robot_joints[i],
                                 posObj=[0, 0, 0],
                                 forceObj=[0, 0, thrusts[i]],
                                 flags=pybullet.LINK_FRAME)

    def apply_external_torques(self, robot, torque):
        pybullet.applyExternalTorque(robot,
                              0,
                              torqueObj=torque,
                              flags=pybullet.LINK_FRAME)
