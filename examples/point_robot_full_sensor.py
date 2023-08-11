import gymnasium as gym
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.scene_examples.obstacles import (
    movable_obstacle,
    dynamicSphereObst3,
)
import numpy as np



def run_point_robot_with_full_sensor(n_steps=10, render=False, obstacles=True, goal=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    #env.add_obstacle(sphereObst1)
    env.add_obstacle(dynamicSphereObst3)
    env.add_obstacle(movable_obstacle)
    env.add_goal(goal1)

    # add sensor
    sensor = FullSensor(['position'], ['position', 'size', 'type'], variance=0.0)
    env.add_sensor(sensor, [0])
    # Set spaces AFTER all components have been added.
    env.set_spaces()
    defaultAction = np.array([0.5, -0.2, 0.0])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    initial_observations = []
    for _ in range(2):
        ob = env.reset(pos=pos0, vel=vel0)
        env.shuffle_goals()
        env.shuffle_obstacles()
        initial_observations.append(ob)
        print(f"Initial observation : {ob}")
        #assert np.array_equal(initial_observations[0], ob)

        history = []
        for _ in range(n_steps):
            action = defaultAction
            ob, *_ = env.step(action)
            for obstacle_index in list(ob['robot_0']['FullSensor']['obstacles'].keys()):
                ob_type = ob['robot_0']['FullSensor']['obstacles'][obstacle_index]['type']
                ob['robot_0']['FullSensor']['obstacles'][obstacle_index]['type'] = "".join([chr(i) for i in ob_type])
            history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot_with_full_sensor(render=True, n_steps=300)
