import gym
import urdfenvs.point_robot_urdf
import numpy as np

obstacles = False
goal = False


def main():
    env = gym.make("pointRobotUrdf-vel-v0", dt=0.05, render=True, flatten_observation=True)
    action = np.array([0.1, 0.0, 1.0])
    n_steps = 100000
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")
    env.add_walls()
    if obstacles:
        from examples.scene_objects.obstacles import (
            sphereObst1,
            sphereObst2,
            urdfObst1,
            dynamicSphereObst3,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(urdfObst1)
        env.add_obstacle(dynamicSphereObst3)
    if goal:
        from examples.scene_objects.goal import splineGoal

        env.add_goal(splineGoal)
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
