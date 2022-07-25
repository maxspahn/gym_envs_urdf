import gym
import urdfenvs.generic_reacher
import numpy as np

goal = False
obstacles = False


def main():
    urdf_file = "ur5.urdf"

    env = gym.make(
        "generic-reacher-tor-v0", dt=0.01, urdf=urdf_file, render=True
    )
    n = env.n()
    action = np.ones(n) * -0.0
    action[0] = 0.0
    n_steps = 100000
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    ob = env.reset(pos=pos0)
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
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
