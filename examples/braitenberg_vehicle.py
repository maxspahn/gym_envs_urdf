import gym
import urdfenvs.braitenberg_robot
from urdfenvs.sensors.light_sensor import LightSensor
import numpy as np

goal = True

def initialize_goal():
    from MotionPlanningGoal.staticSubGoal import StaticSubGoal
    from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal
    goal1Dict = {
        "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
        'desired_position': [0, 0.15, 0.07], 'epsilon': 0.1, 'type': "staticSubGoal", 
    }

    return StaticSubGoal(name="goal1", contentDict=goal1Dict)


def main():
    env = gym.make("braitenberg-robot-vel-v0", dt=0.01, render=True)
    action = np.array([0.8, 0.8])
    n_steps = 1000
    pos0 = np.array([-2, 0, 0])
    light_sensor_1 = LightSensor('light_sensor_1_link')
    #light_sensor_2 = LightSensor('light_sensor_2_link')
    ob = env.reset(pos=pos0)
    if goal:
        env.add_goal(initialize_goal())
    env.add_sensor(light_sensor_1)
    #env.add_sensor(light_sensor_2)
    #print(f"Initial observation : {ob}")
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        print(ob)


if __name__ == "__main__":
    main()
