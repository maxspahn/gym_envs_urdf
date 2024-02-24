import os
import shutil
import numpy as np
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.scene_examples.obstacles import sphereObst1, sphereObst2, wall_obstacles, cylinder_obstacle, dynamicSphereObst2
from urdfenvs.scene_examples.goal import goal1


ROBOTTYPE = 'panda'
ROBOTMODEL = 'panda_without_gripper'


def run_generic_mujoco(n_steps: int = 1000, render: bool = True, goal: bool = False, obstacles: bool = False):
    if goal:
        goal_list = [goal1]
    else:
        goal_list = []
    if obstacles:
        obstacle_list= [sphereObst1, sphereObst2, cylinder_obstacle, dynamicSphereObst2] + wall_obstacles
    else:
        obstacle_list= []
    if os.path.exists(ROBOTTYPE):
        shutil.rmtree(ROBOTTYPE)
    robot_model_original = RobotModel(ROBOTTYPE, ROBOTMODEL)
    robot_model_original.copy_model(os.path.join(os.getcwd(), ROBOTTYPE))
    robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)

    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    env = GenericMujocoEnv(robots, obstacle_list, goal_list, render=render)
    ob, info = env.reset()

    action_mag = np.random.rand(env.nu) * 1.0
    t = 0.0
    history = []
    for _ in range(n_steps):
        action = action_mag * np.cos(env.t)
        ob, _, terminated, _, info = env.step(action)
        history.append(ob)
        if terminated:
            print(info)
            break
    env.close()
    return history

if __name__ == "__main__":
    run_generic_mujoco(n_steps=int(1e3), render=True, obstacles=True, goal=True)
