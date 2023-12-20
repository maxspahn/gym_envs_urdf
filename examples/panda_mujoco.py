import numpy as np
import sys
import gymnasium as gym
from urdfenvs.urdf_common.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.urdf_common.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.scene_examples.obstacles import sphereObst1, sphereObst2, wall_obstacles, cylinder_obstacle


def run_panda(n_steps: int = 1000, render: bool = True, physics_engine: str = 'bullet'):
    obstacles = [sphereObst1, sphereObst2, cylinder_obstacle] + wall_obstacles
    if physics_engine == 'mujoco':
        robots  = [
            GenericMujocoRobot(xml_file='panda_scene.xml'),
        ]
        env = GenericMujocoEnv(robots, obstacles, render=True)
    if physics_engine == 'bullet':
        robots = [
            GenericUrdfReacher(urdf="panda.urdf", mode="vel"),
        ]
        env= UrdfEnv(
            dt=0.01, robots=robots,
            render=True,
            observation_checking=False,
        )
        for obstacle in obstacles:
            env.add_obstacle(obstacle)
        env.set_spaces()


    #observation, info = env.reset(seed=42)
    pos0=np.array([0,0,0,-1.57079,0,1.57079,-0.7853,0.04,0.04])
    env.reset(pos=pos0)

    action_mag = np.array([0.3, -0.2, 0.3, -0.15, 0.2, -0.01, 0.35, 0.01])
    t = 0.0
    for _ in range(n_steps):
        t += env.dt
        action = action_mag * np.cos(t)
        observation, *_ = env.step(action)
    env.close()

if __name__ == "__main__":
    run_panda(n_steps=int(1e8), render=True, physics_engine=sys.argv[1])
