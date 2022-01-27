import gym
import pointRobotUrdf
from sensors.lidar import Lidar
import numpy as np
import cProfile
import threading
import time

num_threads = 2

exitFlag = 0

class myThread(threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter

   def run(self):
      print ("Starting " + self.name)
      run_env(self.name)
      print ("Exiting " + self.name)

def run_env(threadName):
    env = gym.make('pointRobotUrdf-vel-v0', dt=0.05, render=False)
    ob = env.reset()
    for j in range(100):
        print(j)
        ob, *_ = env.step(np.random.random(2) * 0.01)
        print(f"{threadName} at {i} : {ob['x']}")

def print_time(threadName, delay, counter):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print("%s: %s" % ( threadName, time.ctime(time.time()) ))

threads = []
for i in range(num_threads):
    threads.append(myThread(i, f"thread_{i}", 1))

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("Exiting main thread")
