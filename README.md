### Generic URDF robots

In this package, generic urdf robots and a panda gym environment are available.
The goal is to make this environment as easy as possible to deploy. Although, we used the
OpenAI-Gym framing, these environments are not necessarly restricted to
Reinforcement-Learning but rather to local motion planning in general.

<table>
 <tr>
  <td> Point Robot </td>
  <td> Point Robot with Keyboard Input </td>
  <td> Non-Holonomic Robot </td>
 </tr>
 <tr>
  <td> <img src="/docs/source/img/pointRobot.gif" width="250" height="250"/> </td>
  <td> <img src="/docs/source/img/pointRobotKeyboardInput.gif" width="250" height="250"/> </td>  
  <td> <img src="/docs/source/img/boxerRobot.gif" width="250" height="250"/> </td>
 </tr>
</table>

<table>
 <tr>
  <td> Tiago Robot </td>
  <td> Tiago Robot with Keyboard Input </td>
 </tr>
 <tr>
  <td> <img src="/docs/source/img/tiago.gif" width="250" height="250"/> </td>
  <td> <img src="/docs/source/img/tiagoKeyboardInput.gif" width="250" height="250"/> </td>
 </tr>
</table>

<table>
 <tr>
  <td> Panda Robot </td>
  <td> Albert Robot </td>
  </tr>
 <tr>
  <td> <img src="/docs/source/img/panda.gif" width="250" height="250"/> </td>
  <td> <img src="/docs/source/img/albert.gif" width="250" height="250"/> </td>
  </tr>
</table>

Click here for the detailed [documentation](https://maxspahn.github.io/gym_envs_urdf/#).


## Installation

This package depends on casadi for dynamics generation and gym.
Dependencies should be installed through pip or poetry, see below.

Using pip, you can use
```bash
pip3 install '.[options]'
```

Using poetry 
```bash
poetry install -E <options>
```

Options are `keyboard` and `scenes`.

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

An example can be found in examples/keyboard_input_example.py

## Examples

Examples can be found in the corresponding folder.
