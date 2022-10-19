import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import numpy as np
import os

def run_generic_holonomic(n_steps=1000, render=False, goal=True, obstacles=True):
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/ur5.urdf"
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    n = env.n()
    action = np.ones(n) * -0.2
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    mount_positions = np.array([np.array([0.0, 0.0, 1.2])])
    mount_orientations = np.array([np.array([1.0, 0.2, 0.4, 0.0])])
    ob = env.reset(pos=pos0, mount_positions=mount_positions, mount_orientations=mount_orientations)
    print(f"Initial observation : {ob}")
    if goal:
        from examples.scene_objects.goal import dynamicGoal
        env.add_goal(dynamicGoal)

    if obstacles:
        from examples.scene_objects.obstacles import dynamicSphereObst2
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
    run_generic_holonomic(render=True)
