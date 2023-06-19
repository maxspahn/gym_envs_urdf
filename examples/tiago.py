import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
import numpy as np

from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_tiago(n_steps=1000, render=False, goal=True, obstacles=True):
    torso_joint_name = ["torso_lift_joint"]
    head_joint_names = ["head_" + str(i) + "_joint" for i in range(1, 3)]
    arm_right_joint_names = ["arm_right_" + str(i) +
                                "_joint" for i in range(1, 8)]
    arm_left_joint_names = ["arm_left_" + str(i) +
                                "_joint" for i in range(1, 8)]
    actuated_joints = (
        torso_joint_name
        + head_joint_names
        + arm_left_joint_names
        + arm_right_joint_names
    )

    robots = [
        GenericDiffDriveRobot(
            urdf="tiago_dual.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=[
                "caster_front_right_2_joint",
                "caster_front_left_2_joint",
                "caster_back_right_2_joint",
                "caster_back_left_2_joint",
            ],
            not_actuated_joints=[
                "suspension_right_joint",
                "suspension_left_joint",
            ],
            wheel_radius = 0.1,
            wheel_distance = 0.4044,
            spawn_offset = np.array([-0.1764081, 0.0, 0.1]),
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(env.n())
    action[0:2] = np.array([0.2, 0.02])
    action[5] = 0.0 # left arm shoulder
    action[14] = 0.0 # right arm shoulder
    action[22] = 0.01 # finger joint
    vel0 = np.zeros(env.n())
    ob, *_ = env.reset()
    print("base: ", ob['robot_0']["joint_state"]["position"][0:3])
    print("torso: ", ob['robot_0']["joint_state"]["position"][3])
    print("head: ", ob['robot_0']["joint_state"]["position"][4:6])
    print("left arm: ", ob['robot_0']["joint_state"]["position"][6:13])
    print("right arm: ", ob['robot_0']["joint_state"]["position"][13:20])
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_tiago(render=True)
