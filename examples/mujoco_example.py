import os
import shutil
import numpy as np
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.scene_examples.obstacles import sphereObst1, sphereObst2, wall_obstacles, cylinder_obstacle


ROBOTTYPE = 'pointRobot'
ROBOTMODEL = 'pointRobot'


def run_generic_mujoco(n_steps: int = 1000, render: bool = True):
    obstacles = [sphereObst1, sphereObst2, cylinder_obstacle] + wall_obstacles
    if os.path.exists(ROBOTTYPE):
        shutil.rmtree(ROBOTTYPE)
    robot_model_original = RobotModel(ROBOTTYPE, ROBOTMODEL)
    robot_model_original.copy_model(os.path.join(os.getcwd(), ROBOTTYPE))
    robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)

    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    env = GenericMujocoEnv(robots, obstacles, render=render)
    ob, info = env.reset()

    action_mag = np.random.rand(env.nu) * 1.0
    t = 0.0
    obs = np.zeros((n_steps, env.nv))
    actions = np.zeros((n_steps, env.nu))
    for i in range(n_steps):
        t += env.dt
        action = action_mag * np.cos(t)
        ob, _, terminated, _, info = env.step(action)
        obs[i] = ob['robot_0']['joint_state']['velocity']
        actions[i] = action
        if terminated:
            print(info)
            break
    env.close()

if __name__ == "__main__":
    run_generic_mujoco(n_steps=int(1e2), render=True)
