import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
robot = p.loadURDF(fileName="test.urdf")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

n = 7
joints = list(range(1, n+1))

act = np.array([0.01, 0.01, -0.02, 0.0, 0.0, 0.0, 0.0])

n_steps = 1000000
for i in range(n_steps):
    for j in range(n):
        pos = p.getJointState(robot, joints[j])
        p.setJointMotorControl2(robot, joints[j],  controlMode=p.VELOCITY_CONTROL, targetVelocity=act[j])
    time.sleep(0.0001)
    p.stepSimulation()
