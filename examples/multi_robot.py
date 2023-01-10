import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.robots.jackal import JackalRobot
from urdfenvs.robots.boxer import BoxerRobot
from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine

def run_multi_robot(n_steps=1000, render=False, obstacles=False, goal=False):
    physics_engine = BulletPhysicsEngine(render)
    robots = [
        GenericUrdfReacher(physics_engine, urdf="pointRobot.urdf", mode="vel"),
        GenericUrdfReacher(physics_engine, urdf="ur5.urdf", mode="acc"),
        JackalRobot(physics_engine, mode='vel'),
        JackalRobot(physics_engine, mode='vel'),
        JackalRobot(physics_engine, mode='vel'),
        BoxerRobot(physics_engine, mode='vel'),
    ]

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots,
    )
    n = env.n()
    action = np.ones(n) * -0.2
    pos0 = np.zeros(n)
    pos0[1] = -0.0
    ns_per_robot = env.ns_per_robot()
    n_per_robot = env.n_per_robot()
    initial_positions = np.array([np.zeros(n) for n in ns_per_robot])
    for i in range(len(initial_positions)):
        if ns_per_robot[i] != n_per_robot[i]:
            initial_positions[i][0:2] = np.array([0.0, i])
    mount_positions = np.array(
        [
            np.array([0.0, i, 0.0]) for i in range(len(ns_per_robot))
        ]
    )
    ob = env.reset(pos=initial_positions,mount_positions=mount_positions)
    print(f"Initial observation : {ob}")
    if goal:
        from urdfenvs.scene_examples.goal import dynamicGoal
        env.add_goal(dynamicGoal)

    if obstacles:
        from urdfenvs.scene_examples.obstacles import dynamicSphereObst2
        env.add_obstacle(dynamicSphereObst2)

    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_multi_robot(render=True, obstacles=True, goal=True)
