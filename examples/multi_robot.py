import gym
import urdfenvs.generic_urdf_reacher
import numpy as np

goal = False
obstacles = False


def main():
    urdf_files = ["ur5.urdf", "ur5.urdf"]

    env = gym.make(
        "generic-urdf-reacher-v0",
        dt=0.01, urdf=urdf_files, render=True, mode=[3, 2]
    )
    n = env.n()
    action = np.ones(n) * -0.2
    n_steps = 100000
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    ob = env.reset(pos=pos0, base_pos=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]))
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
        print(action)
        ob, _, _, _ = env.step(action)
        print(ob)


if __name__ == "__main__":
    main()
