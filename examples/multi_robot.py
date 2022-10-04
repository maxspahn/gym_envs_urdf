import gym
import numpy as np
from urdfenvs.generic_urdf_reacher.resources.generic_urdf_reacher import GenericUrdfReacher
from urdfenvs.tiago_reacher.resources.tiago_robot import TiagoRobot
from urdfenvs.prius.resources.prius import Prius

goal = True
obstacles = True

def main():
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
    n_steps = 100000
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    base_pos = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -2.0, 0.0]])
    ob = env.reset(pos=pos0, base_pos=base_pos)
    print(f"Initial observation : {ob}")
    if goal:
        from examples.scene_objects.goal import dynamicGoal
        env.add_goal(dynamicGoal)

    if obstacles:
        from examples.scene_objects.obstacles import dynamicSphereObst2
        env.add_obstacle(dynamicSphereObst2)

    print("Starting episode")
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        print(ob)


if __name__ == "__main__":
    main()
