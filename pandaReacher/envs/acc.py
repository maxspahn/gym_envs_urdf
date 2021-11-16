import gym
import numpy as np
import time
import pybullet as p
from pybullet_utils import bullet_client
from pandaReacher.resources.pandaRobot import PandaRobot
from pandaReacher.resources.plane import Plane
from pandaReacher.resources.scene import Scene


class PandaReacherAccEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=True, n=7, dt=0.01, gripper=False):
        print("init")
        self._gripper = gripper
        self._dt = dt
        self.np_random, _ = gym.utils.seeding.np_random()
        self.robot = PandaRobot(gripper, friction=0.02)
        (self.observation_space, self.action_space) = self.robot.getAccSpaces()
        self._isRender = render
        self.clientId = -1
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self._numSubSteps = 20
        self._nSteps = 0
        self._maxSteps = 10000000
        self._p = p
        if self._isRender:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        #self.reset(initialSet=True)
        #self.initSim(timeStep=0.01, numSubSteps=20)

    def dt(self):
        return self._dt

    def addObstacle(self, pos, filename):
        self.robot.addObstacle(pos, filename)


    def step(self, action):
        # Feed action to the robot and get observation of robot's state
        self._nSteps += 1
        self.robot.apply_acc_action(action)
        self._p.stepSimulation()
        ob = self.robot.get_observation()

        # Done by running off boundaries
        reward = 1.0

        if self._nSteps > self._maxSteps:
            reward = reward + 1
            self.done = True
        if self._isRender:
            self.render()
        return ob, reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, q0=np.array([0.0, 0.0, 0.0, -1.501, 0.0, 1.8675, 0.0, 0.0, 0.0]), initialSet=False):
        if not initialSet:
            print("Run " + str(self._nSteps) + " steps in this run")
            self._nSteps = 0
            #self._p.restoreState(self.initState)
            p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.plane = Plane()
        self.Scene = Scene()
        self.robot.reset(poss=q0)
        self.robot.disableVelocityControl()
        self._p.setGravity(0, 0, -9.81)

        p.stepSimulation()

        # Get observation to return
        robot_ob = self.robot.get_observation()

        return robot_ob

    def render(self, mode="none"):
        time.sleep(self.dt())
        return

    def close(self):
        self._p.disconnect()
