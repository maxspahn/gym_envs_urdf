import gym
import numpy as np

from urdfenvs.scene_examples.obstacles import wall_obstacles
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot

def run_jackal(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
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
        ),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.array([1.0, 0.50])
    pos0 = np.array([1.0, 0.2, -1.0])
    ob, info= env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_jackal(render=True, n_steps=1000)
