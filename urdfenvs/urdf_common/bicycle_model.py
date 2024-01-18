import pybullet as p
from typing import List
import gymnasium as gym
import numpy as np
import logging

from urdfenvs.urdf_common.generic_robot import GenericRobot
from urdfenvs.urdf_common.generic_robot import ControlMode


class BicycleModel(GenericRobot):
    """Bicycle model for car like vehicles.

    The bicycle model in velocity mode takes as inputs the forward velocity
    and the steering position. The latter seems counter-intuitive but is more
    common for bicycle models.


    Attributes
    ----------

    _wheel_radius : float
        The radius of the actuated wheels.
    _spawn_offset : np.ndarray
        The offset by which the initial position must be shifted to align
        observation with that position.
    _scaling: float
        The size scaling in which the urdf should be spawned.
        This also effects the dynamics of the system.
    """
    _scaling: float
    _wheel_radius: float
    _wheel_distance: float
    _spawn_offset: np.ndarray
    _facing_direction: str

    def __init__(
            self,
            urdf: str,
            mode: ControlMode,
            actuated_wheels: List[str],
            steering_links: List[str],
            wheel_radius: float,
            wheel_distance: float,
            spawn_offset: np.ndarray = np.array([-0.435, 0.0, 0.05]),
            facing_direction: str = 'x',
            scaling: float = 1.0

        ):
        self._scaling = scaling
        self._wheel_radius = wheel_radius
        self._facing_direction = facing_direction
        self._wheel_distance = wheel_distance
        self._spawn_offset = spawn_offset
        self._steering_links = steering_links
        self._actuated_wheels = actuated_wheels
        self._wheel_radius = wheel_radius
        self._wheel_joints = []
        self._steering_joints = []
        super().__init__(2, urdf, mode)

    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for bycycle models.
        """
        return self.n() + 1

    def extract_joint_ids(self) -> None:
        """Automated extraction of joint ids

        Extract joint ids by the joint names.

        """
        if not hasattr(self, "_steering_joints"):
            return
        if hasattr(self, "_robot"):
            self._robot_joints = []
            self._castor_joints = []
            num_joints = p.getNumJoints(self._robot)
            for name in self._joint_names:
                for i in range(num_joints):
                    joint_info = p.getJointInfo(self._robot, i)
                    joint_name = joint_info[1].decode("UTF-8")
                    link_name = joint_info[12].decode("UTF-8")
                    if joint_name == name:
                        self._robot_joints.append(i)
                    self._link_names.append(link_name)
                    if joint_name in self._actuated_wheels:
                        self._wheel_joints.append(i)
                    if joint_name in self._steering_links:
                        self._steering_joints.append(i)

    def reset(
            self,
            pos: np.ndarray,
            vel: np.ndarray,
            mount_position: np.ndarray,
            mount_orientation: np.ndarray,) -> None:
        """ Reset simulation and add robot """
        logging.warning(
            "The argument 'mount_position' and 'mount_orientation' are \
ignored for bicycle models."
        )
        if hasattr(self, "_robot"):
            p.removeBody(self._robot)
        base_orientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        spawn_position = self._spawn_offset
        spawn_position[0:2] += pos[0:2]
        self._robot = p.loadURDF(
            fileName=self._urdf_file,
            basePosition=spawn_position,
            baseOrientation=base_orientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            globalScaling=self._scaling,
        )
        self.set_joint_names()
        self.extract_joint_ids()
        self.read_limits()
        # set base velocity
        self.update_state()
        self._integrated_forward_velocity = vel[0]

    def read_limits(self) -> None:
        self._limit_pos_j = np.zeros((2, self.ns()))
        self._limit_vel_j = np.zeros((2, self.ns()))
        self._limit_tor_j = np.zeros((2, self.n()))
        self._limit_acc_j = np.zeros((2, self.n()))
        self._limit_pos_steering = np.zeros(2)
        if len(self._steering_joints) > 0:
            joint = self._urdf_robot.robot.joints[self._steering_joints[1] - 1]
            if joint.limit is not None:
                self._limit_pos_steering[0] = joint.limit.lower - 0.1
                self._limit_pos_steering[1] = joint.limit.upper + 0.1
        self._limit_vel_forward_j = np.array([[-40., -10.], [40., 10.]])
        self._limit_pos_j[0, 0:3] = np.array([-1000., -1000., -2 * np.pi])
        self._limit_pos_j[1, 0:3] = np.array([1000., 1000, 2 * np.pi])
        self._limit_vel_j[0, 0:3] = np.array([-40., -40., -10.])
        self._limit_vel_j[1, 0:3] = np.array([40., 40., 10.])
        self.set_acceleration_limits()

    def set_joint_names(self):
        self._joint_names = self._steering_links + self._actuated_wheels

    def set_acceleration_limits(self):
        acc_limit = np.array([1.0, 1.0])
        self._limit_acc_j[0, :] = -acc_limit
        self._limit_acc_j[1, :] = acc_limit

    def check_state_new(self, pos: np.ndarray, vel: np.ndarray):
        if (
            not isinstance(pos, np.ndarray)
            or not pos.size == self.n() + 1
        ):
            pos = np.zeros(self.n() + 1)
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel

    def check_state(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters state of the robot and returns a valid state."""

        if not isinstance(pos, np.ndarray) or not pos.size == self.ns():
            pos = np.zeros(self.ns())
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel

    def get_observation_space(self) -> gym.spaces.Dict:
        """Gets the observation space for a bicycle model.

        The observation space is represented as a dictonary. `x` and `xdot`
        denote the configuration position and velocity, `vel` for forward and
        angular velocity and `steering` is the current steering position.
        """
        return gym.spaces.Dict(
            {
                "joint_state": gym.spaces.Dict(
                    {
                        "position": gym.spaces.Box(
                            low=self._limit_pos_j[0, :],
                            high=self._limit_pos_j[1, :],
                            dtype=float,
                        ),
                        "steering": gym.spaces.Box(
                            low=self._limit_pos_steering[0],
                            high=self._limit_pos_steering[1],
                            shape=(1,),
                            dtype=float,
                        ),
                        "velocity": gym.spaces.Box(
                            low=self._limit_vel_j[0, :],
                            high=self._limit_vel_j[1, :],
                            dtype=float,
                        ),
                        "forward_velocity": gym.spaces.Box(
                            low=self._limit_vel_forward_j[0, :],
                            high=self._limit_vel_forward_j[1, :],
                            dtype=float,
                        ),
                    }
                )
            }
        )

    def get_velocity_spaces(self) -> tuple:
        """Get observation space and action space when using velocity
        control."""
        ospace = self.get_observation_space()
        uu = self._limit_vel_forward_j[1, :]
        ul = self._limit_vel_forward_j[0, :]
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=float)
        return (ospace, aspace)

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies velocities to forward motion and sets steering angle."""
        for steering_joint in self._steering_joints:
            p.setJointMotorControl2(
                self._robot,
                steering_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=vels[1],
            )
        for joint in self._wheel_joints:
            p.setJointMotorControl2(
                self._robot,
                joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[0] / (self._wheel_radius * self._scaling),
            )

    def apply_acceleration_action(self, accs: np.ndarray, dt: float) -> None:
        """Applies acceleration action to the robot.

        The acceleration action relies on integration of the velocity signal
        and applies velocities at the end. The integrated velocities are
        clipped to avoid very large velocities.
        """
        self._integrated_forward_velocity += dt * accs[0]
        actions = np.array([self._integrated_forward_velocity, accs[1]])
        self.apply_velocity_action(actions)

    def apply_torque_action(self, torques: np.ndarray) -> None:
        raise NotImplementedError("Torque action is not available for prius.")

    def correct_base_orientation(self, pos_base: np.ndarray) -> np.ndarray:
        """Corrects base orientation to be within the interval (-pi , pi].
        """
        if self._facing_direction == '-y':
            pos_base[2] -= np.pi/2
        elif self._facing_direction == 'y':
            pos_base[2] += np.pi/2
        elif self._facing_direction == '-x':
            pos_base[2] += np.pi
        if pos_base[2] < -np.pi:
            pos_base[2] += 2 * np.pi
        if pos_base[2] > np.pi:
            pos_base[2] -= 2 * np.pi
        return pos_base

    def update_state(self) -> None:
        """Updates the robot state.

        The robot state is stored in the dictonary self.state.  There, the key
        x refers to the translational and rotational position in the world
        frame. The key xdot refers to the translation and rotational velocity
        in the world frame. The key vel is the current forward and angular
        velocity. For a bicycle model we additionally store information about
        the sterring behind the key steering.
        """
        # base position
        link_state = p.getLinkState(self._robot, 0, computeLinkVelocity=1)
        pos_base = np.array(
            [
                link_state[0][0],
                link_state[0][1],
                p.getEulerFromQuaternion(link_state[1])[2],
            ]
        )
        self.correct_base_orientation(pos_base)
        vel_base = np.array(
            [link_state[6][0], link_state[6][1], link_state[7][2]]
        )
        # wheel velocities
        vel_wheels = p.getJointStates(self._robot, self._wheel_joints[2:4])
        v_right = vel_wheels[0][1]
        v_left = vel_wheels[1][1]
        vel = np.array(
            [
                0.5 * (v_right + v_left) * self._scaling * self._wheel_radius,
                vel_base[2],
            ]
        )
        pos, _, _, _ = p.getJointState(self._robot, self._steering_joints[1])
        steering_pos = np.array([pos])
        self.state = {
            "joint_state": {
                "position": pos_base,
                "forward_velocity": vel,
                "velocity": vel_base,
                "steering": steering_pos,
            }
        }
