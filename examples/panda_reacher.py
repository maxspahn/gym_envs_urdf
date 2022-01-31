import gym
import pandaReacher
import numpy as np

goal = False
obstacles = False


def main():
    gripper = True
    env = gym.make('panda-reacher-vel-v0', dt=0.01, render=True, gripper=gripper)
    defaultAction = np.ones(9) * 0.0
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset()
        if goal:
            from urdfGymExamples.sceneObjects.goal import dynamicGoal
            env.addGoal(dynamicGoal)
        if obstacles:
            from urdfGymExamples.sceneObjects.obstacles import dynamicSphereObst2
            env.addObstacle(dynamicSphereObst2)
        print("Starting episode")
        for i in range(n_steps):
            if (int(i/70))%2 == 0:
                defaultAction[7] = -0.02
                defaultAction[8] = -0.02
            else:
                defaultAction[7] = 0.02
                defaultAction[8] = 0.02
            action = env.action_space.sample()
            if gripper:
                action = defaultAction[0:9]
            else:
                action = defaultAction[0:7]
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    main()
