import gym
import urdfenvs.panda_reacher
import numpy as np

def run_panda(n_steps=1000, render=False, goal=True, obstacles=True):
    gripper = True
    env = gym.make(
        "panda-reacher-vel-v0", dt=0.01, render=render, gripper=gripper
    )
    action = np.ones(9) * 0.0
    ob = env.reset()
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
        history.append(ob)
    return history


if __name__ == "__main__":
    run_panda(render=True)
