import gym
import urdfenvs.braitenberg_robot
import numpy as np

goal = True

def initialize_goal():
    from MotionPlanningGoal.staticSubGoal import StaticSubGoal
    from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal
    goal1Dict = {
        "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
        'desired_position': [3, 2, 0.2], 'epsilon': 0.1, 'type': "staticSubGoal", 
    }

    return StaticSubGoal(name="goal1", contentDict=goal1Dict)


def main():
    env = gym.make("braitenberg-robot-vel-v0", dt=0.01, render=True)
    action = np.array([0.8, 0.8])
    n_steps = 100000
    pos0 = np.array([1.0, 0.2, -1.0])
    ob = env.reset(pos=pos0)
    if goal:
        env.add_goal(initialize_goal())
    print(f"Initial observation : {ob}")
    for _ in range(n_steps):
        ob, *_ = env.step(action)


if __name__ == "__main__":
    main()
