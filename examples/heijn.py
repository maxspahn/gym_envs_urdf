import gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine
import numpy as np

def run_heijn_robot(n_steps=1000, render=False, goal=True, obstacles=True):
    physics_engine = BulletPhysicsEngine(render)
    robots = [
        GenericUrdfReacher(physics_engine, urdf="heijn_robot.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        physics_engine=physics_engine,
        dt=0.01, robots=robots, render=render
    )
    action = np.array([0.1, 0.0, 0.01])
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")
    env.add_walls()
    if obstacles:
        from urdfenvs.scene_examples.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst3,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst3)
    if goal:
        from urdfenvs.scene_examples.goal import splineGoal

        env.add_goal(splineGoal)
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_heijn_robot(render=True)
