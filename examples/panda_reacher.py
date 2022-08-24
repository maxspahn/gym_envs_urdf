import gym
import urdfenvs.panda_reacher # pylint: disable=unused-import
import numpy as np

from examples.scene_objects.goal import dynamicGoal
from examples.scene_objects.obstacles import dynamicSphereObst2

goal = False
obstacles = False


def main():
    gripper = True
    env = gym.make(
        "panda-reacher-vel-v0", dt=0.01, render=True, gripper=gripper
    )
    action = np.ones(9) * 0.0
    n_steps = 100000
    ob = env.reset()
    print(f"Initial observation : {ob}")
    if goal:
        env.add_goal(dynamicGoal)

    if obstacles:
        env.add_goal(dynamicGoal)

    if obstacles:
        env.add_obstacle(dynamicSphereObst2)

    print("Starting episode")
    for i in range(n_steps):
        if (int(i / 70)) % 2 == 0:
            action[7] = -0.02
            action[8] = -0.02
        else:
            action[7] = 0.02
            action[8] = 0.02
        if gripper:
            ob, _, _, _ = env.step(action)
        else:
            ob, _, _, _ = env.step(action[0:7])


if __name__ == "__main__":
    main()
