from typing import List
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.scene_examples.goal import dynamicGoal
from urdfenvs.scene_examples.obstacles import dynamicSphereObst2
from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_panda_capsules(n_steps=100000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="panda_collision_links.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots,
        render=render,
        observation_checking=False,
    )
    env.add_goal(dynamicGoal)
    env.add_obstacle(dynamicSphereObst2)
    env.set_spaces()
    action = np.ones(env.n()) * 0.1
    ob = env.reset()

    link_translations = [
            np.array([0.0, 0.0, -0.1915]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.145]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.26]),
            np.array([0, 0.08, -0.13]),
            np.array([0, 0.0, -0.03]),
            np.array([0, 0.0, 0.01]),
            np.array([0.0424, 0.0424, -0.0250]),
    ]
    link_rotations = [np.identity(3)] * 9
    link_rotations[8] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    collision_links = [0, 1, 2, 3, 4, 4, 5, 6, 7]
    collision_links: List[str] = [
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
    ]
    lengths = [0.2830, 0.12, 0.15, 0.12, 0.1, 0.14, 0.08, 0.14, 0.01]

    radii = [0.09, 0.09, 0.09, 0.09, 0.09, 0.055, 0.08, 0.07, 0.06]
    for i in range(len(collision_links)):
        link_transformation = np.identity(4)
        link_transformation[0:3, 3] = link_translations[i]
        link_transformation[0:3, 0:3] = link_rotations[i]
        env.add_collision_link(
                robot_index=0,
                link_index=collision_links[i],
                shape_type="capsule",
                size=[radii[i], lengths[i]],
                link_transformation=link_transformation,
        )

    print(f"Initial observation : {ob}")
    print("Starting episode")
    history = []
    for i in range(n_steps):
        collision_links_position: dict = env.collision_links_poses(position_only=True)
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_panda_capsules(render=True)
