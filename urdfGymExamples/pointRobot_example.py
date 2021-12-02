import gym
import pointRobotUrdf
from sensors.lidar import Lidar
import numpy as np

obstacles = False
goal = False

def main():
    env = gym.make('pointRobotUrdf-acc-v0', dt=0.05, render=True)
    lidar = Lidar(4, nbRays=4)
    env.addSensor(lidar)
    defaultAction = np.array([0.1, 0.0])
    n_episodes = 1
    n_steps = 100000
    cumReward = 0.0
    pos0 = np.array([1.0, 0.1])
    vel0 = np.array([1.0, 0.0])
    for e in range(n_episodes):
        ob = env.reset(pos=pos0, vel=vel0)
        env.setWalls(limits=[[-3, -2], [3, 2]])
        if obstacles:
            from urdfGymExamples.obstacles import sphereObst1, sphereObst2, urdfObst1

            env.addObstacle(sphereObst1)
            env.addObstacle(sphereObst2)
            env.addObstacle(urdfObst1)
        if goal:
            from urdfGymExamples.goal import goal1

            env.addGoal(goal1)
        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t += env.dt()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    main()
