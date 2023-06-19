import gymnasium as gym
from urdfenvs.scene_examples.obstacles import wall_obstacles
import numpy as np

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot


def run_boxer(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="boxer.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation=np.pi/2,
        ),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.6, 0.8])
    pos0 = np.array([1.0, 0.2, 0.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        print(ob['robot_0']['joint_state']['velocity'][0:2])
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_boxer(render=True)
