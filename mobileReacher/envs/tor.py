import gym
import numpy as np
import time
import pybullet as p
from pybullet_utils import bullet_client
from mobileReacher.resources.mobileRobot import MobileRobot
from mobileReacher.resources.plane import Plane
import matplotlib.pyplot as plt


class MobileReacherTorEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, dt=0.01, gripper=False):
        print("init")
        self._n = 10
        self._gripper = gripper
        self._dt = dt
        self.np_random, _ = gym.utils.seeding.np_random()
        self.robot = MobileRobot(gripper=gripper)
        (self.observation_space, self.action_space) = self.robot.getTorqueSpaces()
        self._isRender = render
        self.clientId = -1
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self._numSubSteps= 20
        self._nSteps = 0
        self._maxSteps = 10000000
        self._p = p
        if self._isRender:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.reset(initialSet=True)
        #self.initSim(timeStep=0.01, numSubSteps=20)

    def dt(self):
        return self._dt

    def step(self, action):
        # Feed action to the robot and get observation of robot's state
        self._nSteps += 1
        if self._gripper:
            self.robot.apply_torque_action(action[:-1])
            self.robot.moveGripper(action[-1])
        else:
            self.robot.apply_torque_action(action)
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

    def initSim(self, numSubSteps):
        if self.isRender:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self.clientId = self._p._client

        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=numSubSteps
        )
        self._p.setGravity(0, 0, -10)
        # Load the plane and robot
        self.plane = Plane(self.clientId)
        #self.robot = GenericReacherMobile(self.clientId)
        self.done = False

        # Visual element of the goal
        self.initState = self._p.saveState()

    def reset(self, initialSet=False):
        if not initialSet:
            print("Run " + str(self._nSteps) + " steps in this run")
            self._nSteps = 0
            #self._p.restoreState(self.initState)
            p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.plane = Plane()
        self.robot.reset()
        self.robot.disableVelocityControl()
        self._p.setGravity(0, 0, -10)

        p.stepSimulation()

        # Get observation to return
        robot_ob = self.robot.get_observation()

        return robot_ob

    def render(self, mode="none"):
        self.dt()
        return
        """
        if mode == "human":
            self.isRender = True
        else:
            self.isRender = False
        if self.initialRender:
            self.initialRender = False
            self.initSim(timeStep=0.02, numSubSteps=20)
        """

    def close(self):
        self._p.disconnect()
