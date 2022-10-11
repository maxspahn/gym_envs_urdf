import gym
import numpy as np
from urdfenvs.robots.iris import IrisDrone


def run_iris(n_steps=3000, render=False, goal=True, obstacles=True):
    robots = [
        IrisDrone(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # set initial state [x, y, z, qx, qy, qz, qw]
    pos0 = np.array([-2.0, 0.0, 1.2, 0., 0.0, 0.0, 1.])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    history = []
    for i in range(n_steps):
        action = np.ones(4) * 830.2 # relatively stable motion
        if i > 200 and i < 248:
            action += np.array([0, 1, 0, 1]) * 1
        if i > 248 and i < 298:
            action += np.array([0, 1, 0, 1]) * -1
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history
        

if __name__ == "__main__":
    run_iris(render=True)
