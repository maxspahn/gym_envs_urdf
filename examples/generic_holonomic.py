import gym
import urdfenvs.generic_urdf_reacher # pylint: disable=unused-import
import numpy as np

from examples.scene_objects.goal import dynamicGoal
from examples.scene_objects.obstacles import dynamicSphereObst2

goal = False
obstacles = False


def main():
    urdf_file = "ur5.urdf"

    env = gym.make(
        "generic-urdf-reacher-vel-v0", dt=0.01, urdf=urdf_file, render=True
    )
    n = env.n()
    action = np.ones(n) * -0.2
    n_steps = 100000
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    if goal:
        env.add_goal(dynamicGoal)

    if obstacles:
        env.add_goal(dynamicGoal)

    if obstacles:

        env.add_obstacle(dynamicSphereObst2)
    print("Starting episode")
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
