import gym
from urdfenvs.robots.prius import Prius
import numpy as np

from urdfenvs.urdf_common.bullet_physics_engine import BulletPhysicsEngine


def run_prius(n_steps=1000, render=False, goal=True, obstacles=True):
    physics_engine = BulletPhysicsEngine(render)
    robots = [
        Prius(physics_engine, mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        physics_engine=physics_engine,
        dt=0.01, robots=robots, render=render
    )
    action = np.array([1.1, 0.1])
    pos0 = np.array([-1.0, 0.2, -1.0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if ob['robot_0']['joint_state']['steering'] > 0.2:
            action[1] = 0
        history.append(ob)
    env.close()
    return history

if __name__ == "__main__":
    run_prius(render=True)
