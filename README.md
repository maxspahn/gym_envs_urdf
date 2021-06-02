### Generic Planar Robots

In this package, generic urdf robots and a panda gym environment are available.

## Dependencies

This package depends on casadi for dynamics generation and gym.
Dependencies should be installed through pip installation, see below.

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

## Examples

Simple examples can be found in the corresponding folder.
