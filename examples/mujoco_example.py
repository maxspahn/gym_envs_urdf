import os
import time
import shutil
from typing import Union
import numpy as np
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.sensors.free_space_decomposition import FreeSpaceDecompositionSensor
from urdfenvs.sensors.free_space_occupancy import FreeSpaceOccupancySensor
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.lidar import Lidar
from urdfenvs.sensors.sdf_sensor import SDFSensor
from urdfenvs.scene_examples.obstacles import sphereObst1, sphereObst2, wall_obstacles, cylinder_obstacle, dynamicSphereObst1
from urdfenvs.scene_examples.goal import goal1
from gymnasium.wrappers import RecordVideo


ROBOTTYPE = 'panda'
ROBOTMODEL = 'panda_without_gripper'
ROBOTTYPE = 'pointRobot'
ROBOTMODEL = 'pointRobot'

def get_index_from_coordinates(point, mesh) -> tuple:
    distances = np.linalg.norm(mesh - point, axis=3)
    return np.unravel_index(np.argmin(distances), mesh.shape[:-1])

def evaluate_sdf(point, mesh, sdf, resolution) -> tuple:
    index = list(get_index_from_coordinates(point, mesh))
    value = sdf[tuple(index)]
    gradient = np.zeros(3)
    for dim in range(3):
        lower_index = tuple(index[:dim] + [index[dim] - 1] + index[dim+1:])
        upper_index = tuple(index[:dim] + [index[dim] + 1] + index[dim+1:])

        if lower_index[dim] < 0 or upper_index[dim] >= sdf.shape[dim]:
            gradient[dim] = 0.0
        else:
            gradient[dim] = (sdf[upper_index] - sdf[lower_index]) / resolution[dim]

    return value, gradient


def run_generic_mujoco(n_steps: int = 1000, render: Union[str, bool] = True, goal: bool = False, obstacles: bool = False):
    if goal:
        goal_list = [goal1]
    else:
        goal_list = []
    if obstacles:
        obstacle_list= [sphereObst1, sphereObst2, cylinder_obstacle, dynamicSphereObst1] + wall_obstacles
    else:
        obstacle_list= []
    full_sensor = FullSensor(['position'], ['position', 'size', 'type', 'orientation'], variance=0.0, physics_engine_name='mujoco')
    sdf_sensor = SDFSensor(
        limits =  np.array([[-2, 2], [-2, 2], [0, 0]]),
        resolution = np.array([101, 101, 1], dtype=int),
        interval=10,
        physics_engine_name='mujoco',
    )
    val = 40
    fsd_sensor = FreeSpaceOccupancySensor(
        'base_link',
        plotting_interval=-1,
        plotting_interval_fsd=-1,
        max_radius=10,
        number_constraints=10,
        limits =  np.array([[-5, 5], [-5, 5], [0, 50/val]]),
        resolution = np.array([val + 1, val + 1, 5], dtype=int),
        interval=100,
        physics_engine_name='mujoco',
    )
    number_lidar_rays = 64
    lidar_sensor = Lidar(
        "base_link",
        nb_rays=number_lidar_rays,
        ray_length=5.0,
        raw_data=False,
        physics_engine_name='mujoco',
    )
    free_space_decomp = FreeSpaceDecompositionSensor(
        "base_link",
        nb_rays=number_lidar_rays,
        max_radius=10,
        number_constraints=1,
        physics_engine_name='mujoco',
    )
    if os.path.exists(ROBOTTYPE):
        shutil.rmtree(ROBOTTYPE)
    robot_model_original = RobotModel(ROBOTTYPE, ROBOTMODEL)
    robot_model_original.copy_model(os.path.join(os.getcwd(), ROBOTTYPE))
    robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)

    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    env = GenericMujocoEnv(
        robots,
        obstacle_list,
        goal_list,
        sensors=[lidar_sensor, full_sensor, free_space_decomp, sdf_sensor],
        render=render,
    )
    trigger = lambda t: t % 200 == 0
    action_mag = np.random.rand(env.nu) * 1.0
    if render == 'rgb_array':
        env = RecordVideo(env, video_folder=f'{ROBOTMODEL}.mp4')
    ob, info = env.reset()

    t = 0.0
    history = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        action = action_mag * np.cos(i/20)
        ob, _, terminated, _, info = env.step(action)
        #print(ob['robot_0'])
        history.append(ob)
        if terminated:
            print(info)
            break
        t1 = time.perf_counter()
        print(f"actual time: {t1-t0}")

    env.close()
    return history

if __name__ == "__main__":
    run_generic_mujoco(n_steps=int(1e3), render='human', obstacles=True, goal=True)
