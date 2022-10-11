Generic URDF robots
===================

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/maxspahn/gym_envs_urdf.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/maxspahn/gym_envs_urdf/context:python)

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
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/pointRobot.gif" width="250" height="250"/> </td>
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/pointRobotKeyboardInput.gif" width="250" height="250"/> </td>  
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/boxerRobot.gif" width="250" height="250"/> </td>
 </tr>
</table>

<table>
 <tr>
  <td> Tiago Robot </td>
  <td> Tiago Robot with Keyboard Input </td>
 </tr>
 <tr>
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/tiago.gif" width="250" height="250"/> </td>
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/tiagoKeyboardInput.gif" width="250" height="250"/> </td>
 </tr>
</table>

<table>
 <tr>
  <td> Panda Robot </td>
  <td> Albert Robot </td>
  </tr>
 <tr>
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/panda.gif" width="250" height="250"/> </td>
  <td> <img src="https://raw.githubusercontent.com/maxspahn/gym_envs_urdf/master/docs/source/img/albert.gif" width="250" height="250"/> </td>
  </tr>
</table>

Getting Started
================

This is the guide to quickle get going with urdf gym environments.

Pre-requisites
==============

-   Python &gt;3.6, &lt;3.10
-   pip3
-   git

Installation
============

You first have to downlad the repository

``` {.sourceCode .bash}
git clone git@github.com:maxspahn/gym_envs_urdf.git
```

Then, you can install the package using pip as:

``` {.sourceCode .bash}
pip3 install .
```

Optional: Installation with poetry
==================================

If you want to use [poetry](https://python-poetry.org/docs/), you have
to install it first. See their webpage for instructions
[docs](https://python-poetry.org/docs/). Once poetry is installed, you
can install the virtual environment with the following commands. Note
that during the first installation `poetry update` takes up to 300 secs.

``` {.sourceCode .bash}
poetry install
```

The virtual environment is entered by

``` {.sourceCode .bash}
poetry shell
```

Inside the virtual environment you can access all the examples.

Examples
========

Run example
-----------

You find several python scripts in
[examples/](https://github.com/maxspahn/gym_envs_urdf/tree/master/examples).
You can test those examples using the following (if you use poetry, make
sure to enter the virtual environment first with `poetry shell`)

``` {.sourceCode .python}
python3 point_robot.py
```

Replace point_robot.py with the name of the script you want to run.

Use environments
----------------

In the `examples`, you will find individual examples for all implemented
robots. Environments can be created using the normal gym syntax. Gym
environments rely mostly on three functions

-   `gym.make(...)` to create the environment,
-   `gym.reset(...)` to reset the environment,
-   `gym.step(action)` to step one time step in the environment.

For example, in
[examples/point_robot.py](https://github.com/maxspahn/gym_envs_urdf/blob/master/examples/point_robot.py),
you can find the following syntax to `make`, `reset` and `step` the
environment.

``` {.sourceCode .python}
env = gym.make('pointRobotUrdf-vel-v0', dt=0.05, render=True)
ob = env.reset(pos=pos0, vel=vel0)
ob, reward, done, info = env.step(action)
```

The id-tag in the `make` command specifies the robot and the control
type. You can get a full list of all available environments using

``` {.sourceCode .python}
from gym import envs
print(envs.registry.all())
```

Go ahead and explore all the examples you can find there.
