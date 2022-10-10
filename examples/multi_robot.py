import gym
import numpy as np
from urdfenvs.robots.tiago import TiagoRobot
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.robots.prius import Prius

def run_multi_robot(n_steps=1000, render=False, obstacle=False, goal=False):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        # GenericUrdfReacher(urdf="ur5.urdf", mode="acc"),
        # GenericUrdfReacher(urdf="ur5.urdf", mode="acc"),
        TiagoRobot(mode="vel"),
        Prius(mode="vel")
    ]

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=True
    )
    n = env.n()
    action = np.ones(n) * -0.2
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    base_pos = np.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -2.0, 0.0]
        ])
    ob = env.reset(pos=pos0, base_pos=base_pos)
    print(f"Initial observation : {ob}")
    if goal:
        from examples.scene_objects.goal import dynamicGoal
        env.add_goal(dynamicGoal)

    if obstacles:
        from examples.scene_objects.obstacles import dynamicSphereObst2
        env.add_obstacle(dynamicSphereObst2)

    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_multi_robot(render=True)
