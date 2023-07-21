import gymnasium as gym
import os
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.robots.generic_urdf import GenericDiffDriveRobot

def run_multi_robot(n_steps=1000, render=False, obstacles=False, goal=False):
    jackal_1 = GenericDiffDriveRobot(
        urdf="jackal.urdf",
        mode="vel",
        actuated_wheels=[
            "rear_right_wheel",
            "rear_left_wheel",
            "front_right_wheel",
            "front_left_wheel",
        ],
        castor_wheels=[],
        wheel_radius = 0.098,
        wheel_distance = 2 * 0.187795 + 0.08,
    )
    jackal_2 = GenericDiffDriveRobot(
        urdf="jackal.urdf",
        mode="vel",
        actuated_wheels=[
            "rear_right_wheel",
            "rear_left_wheel",
            "front_right_wheel",
            "front_left_wheel",
        ],
        castor_wheels=[],
        wheel_radius = 0.098,
        wheel_distance = 2 * 0.187795 + 0.08,
    )
    boxer = GenericDiffDriveRobot(
        urdf="boxer.urdf",
        mode="vel",
        actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
        castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
        wheel_radius = 0.08,
        wheel_distance = 0.494,
    )
    ur5_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/ur5.urdf"
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf=ur5_urdf_file, mode="acc"),
        jackal_1,
        jackal_2,
        boxer,
    ]

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    n = env.n()
    action = np.ones(n) * -0.2
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    ns_per_robot = env.ns_per_robot()
    n_per_robot = env.n_per_robot()
    initial_positions = np.array([np.zeros(n) for n in ns_per_robot])
    for i in range(len(initial_positions)):
        if ns_per_robot[i] != n_per_robot[i]:
            initial_positions[i][0:2] = np.array([0.0, i])
    mount_positions = np.array(
        [
            np.array([0.0, i, 0.0]) for i in range(len(ns_per_robot))
        ]
    )
    ob = env.reset(pos=initial_positions,mount_positions=mount_positions)
    print(f"Initial observation : {ob}")
    if goal:
        from urdfenvs.scene_examples.goal import dynamicGoal
        env.add_goal(dynamicGoal)

    if obstacles:
        from urdfenvs.scene_examples.obstacles import dynamicSphereObst2
        env.add_obstacle(dynamicSphereObst2)

    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_multi_robot(render=True, obstacles=True, goal=True)
