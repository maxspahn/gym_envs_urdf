### Generic Planar Robots

In this package, generic urdf robots and a panda gym environment are available.

## Dependencies

This package depends on casadi for dynamics generation and gym.
Dependencies should be installed through pip installation, see below.
When using obstacles and goals, this package also depends on MotionPlanningScenes,
see https://github.com/maxspahn/motion_planning_scenes

You can either install that manually as explained in the corresponding readme or 
you install it through
```bash
pip3 install -r requirements_scenes.py
```

## Installation
```bash
pip3 install -e .
```

## Switching

Environments can be created using the normal gym syntax.
For example the below code line creates a planar robot with 3 links and a constant k.
Actions are torques to the individual joints.
```python
env = gym.make('nLink-urdf-reacher-vel-v0', n=3, dt=0.01, render=True)
```

A holonomic and a differential drive mobile manipulator are implemented:
```python
env = gym.make('albert-reacher-vel-v0', dt=0.01, render=True)
env = gym.make('mobile-reacher-tor-v0', dt=0.01, render=True)
```
For most robots, different control interfaces are available, velocity control,
acceleration control and torque control.

## Robot control with the keyboard 
Control robot actuators with keyboard keys. This is done by:
* setting up a parent en child process with a pipe connection inbetween
* setup and start main process with parent_connection as arguement
* setup Responder object with child_connection as arguement
* start Responder with parent process as arguement

In the main loop an request for action should be made followed by wainting
for a response as such:
```python
parent_conn.send({"request_action": True})
keyboard_data = parent_conn.recv()
action = keyboard_data["action"]
```
this feature requires extra dependencies
```bash
pip3 install -e .[keyboard_input]
```

An example can be found in ./urdfGymExamples/keyboard_input_example.py


## Examples

Simple examples can be found in the corresponding folder.
