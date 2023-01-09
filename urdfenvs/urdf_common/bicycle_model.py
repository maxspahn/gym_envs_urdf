import gym
import numpy as np
import logging

from urdfenvs.urdf_common.generic_robot import GenericRobot
from urdfenvs.urdf_common.utils import euler_to_quat


class BicycleModel(GenericRobot):
    """Bicycle model for car like vehicles.

    Attributes
    ----------

    _wheel_radius : float
        The radius of the actuated wheels.
    _spawn_offset : np.ndarray
        The offset by which the initial position must be shifted to align
        observation with that position.
    """

    def __init__(self, physics_engine, n: int, urdf_file: str, mode: str):
        """Constructor for bicyle model robot."""
        super().__init__(physics_engine, n, urdf_file, mode)
        self._wheel_radius: float = None
        self._spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15])

    def ns(self) -> int:
        """Returns the number of degrees of freedom.

        This is needed as number of actuated joints `_n` is lower that the
        number of degrees of freedom for bycycle models.
        """
        return self.n() + 1

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
            self._physics_engine.reset_simulation()
        base_orientation = euler_to_quat(0, 0, pos[0])
        spawn_position = self._spawn_offset
        spawn_position[0:2] += pos[0:2]
        self._robot = self._physics_engine.load_urdf(
            self._urdf_file,
            spawn_position,
            base_orientation,
            scaling=self._scaling,
        )
        self.set_joint_names()
        self.extract_joint_ids()
        self.read_limits()
        # set base velocity
        self.update_state()
        self._integrated_velocities = vel

    def read_limits(self) -> None:
        self._limit_pos_j = np.zeros((2, self.ns()))
        self._limit_vel_j = np.zeros((2, self.ns()))
        self._limit_tor_j = np.zeros((2, self.n()))
        self._limit_acc_j = np.zeros((2, self.n()))
        self._limit_pos_steering = np.zeros(2)
        joint = self._urdf_robot.robot.joints[self._steering_joints[1] - 1]
        self._limit_pos_steering[0] = joint.limit.lower - 0.1
        self._limit_pos_steering[1] = joint.limit.upper + 0.1
        self._limit_vel_forward_j = np.array([[-40., -10.], [40., 10.]])
        self._limit_pos_j[0, 0:3] = np.array([-1000., -1000., -2 * np.pi])
        self._limit_pos_j[1, 0:3] = np.array([1000., 1000, 2 * np.pi])
        self._limit_vel_j[0, 0:3] = np.array([-40., -40., -10.])
        self._limit_vel_j[1, 0:3] = np.array([40., 40., 10.])
        self.set_acceleration_limits()

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
                            dtype=np.float64,
                        ),
                        "steering": gym.spaces.Box(
                            low=self._limit_pos_steering[0],
                            high=self._limit_pos_steering[1],
                            shape=(1,),
                            dtype=np.float64,
                        ),
                        "velocity": gym.spaces.Box(
                            low=self._limit_vel_j[0, :],
                            high=self._limit_vel_j[1, :],
                            dtype=np.float64,
                        ),
                        "forward_velocity": gym.spaces.Box(
                            low=self._limit_vel_forward_j[0, :],
                            high=self._limit_vel_forward_j[1, :],
                            dtype=np.float64,
                        ),
                    }
                )
            }
        )

    def apply_velocity_action(self, vels: np.ndarray) -> None:
        """Applies velocities to steering and forward motion."""
        self._physics_engine.apply_velocity_action(
            vels[1:2],
            self._robot, 
            self._steering_joints[1:2],
        )
        pos_wheel_right, _ = self._physics_engine.joint_states(
            self._robot,
            self._steering_joints[1:2]
        )
        self._physics_engine.apply_position_action(
            pos_wheel_right,
            self._robot,
            self._steering_joints[0:1],
        )
        forward_velocity = vels[0] / (self._wheel_radius * self._scaling)
        self._physics_engine.apply_velocity_action(
            np.array([forward_velocity, forward_velocity]),
            self._robot,
            self._forward_joints,
        )

    def apply_acceleration_action(self, accs: np.ndarray, dt: float) -> None:
        raise NotImplementedError("Acceleration action is not available for prius.")

    def apply_torque_action(self, torques: np.ndarray) -> None:
        raise NotImplementedError("Torque action is not available for prius.")

    def correct_base_orientation(self, pos_base: np.ndarray) -> np.ndarray:
        """Corrects base orientation by -pi.

        The orientation observation should be zero when facing positive
        x-direction. Some robot models are rotated by pi. This is corrected
        here. The function also makes sure that the orientation is always
        between -pi and pi.
        """
        pos_base[2] -= np.pi
        if pos_base[2] < -np.pi:
            pos_base[2] += 2 * np.pi
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

        base_position, wheel_velocity= self._physics_engine.get_base_state(self._robot, self._robot_joints, self.correct_base_orientation)
        v_right = wheel_velocity[0]
        v_left = wheel_velocity[1]
        # simple dynamics model to compute the forward and angular velocity
        vel_base = self._physics_engine.get_link_state(self._robot, 0)[7][2]
        velocity_base = np.array(
            [
                0.5 * (v_right + v_left) * self._scaling * self._wheel_radius,
                vel_base,
            ]
        )

        # joint configurations for holonomic joints
        steering_pos, _= self._physics_engine.joint_states(self._robot, self._steering_joints[1:2])


        self.state = {
            "joint_state": {
                "position": base_position,
                "forward_velocity": velocity_base,
                "velocity": velocity_base,
                "steering": steering_pos,
            }
        }
