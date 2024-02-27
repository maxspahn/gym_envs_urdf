Generic URDF robots
===================

In this package, generic urdf robots and a panda gym environment are
available. The goal is to make this environment as easy as possible to
deploy. Although, we used the OpenAI-Gym framing, these environments are
not necessarly restricted to Reinforcement-Learning but rather to local
motion planning in general.

Pybullet
--------

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

Mujoco
--------

<table>
 <tr>
  <td> Point Robot </td>
  <td> Panda Robot </td>
  </tr>
 <tr>
  <td> <img src="/docs/source/img/pointRobot_mujoco.gif" width="250" height="250"/> </td>
  <td> <img src="/docs/source/img/panda_without_gripper.gif" width="250" height="250"/> </td>
  </tr>
</table>

Getting started
===============

This is the guide to quickle get going with urdf gym environments.

Pre-requisites
--------------

-   Python \>=3.8
-   pip3
-   git

Installation from pypi
----------------------

The package is uploaded to pypi so you can install it using

``` {.sourceCode .bash}
pip3 install urdfenvs
```

Installation from source
------------------------

You first have to download the repository

``` {.sourceCode .bash}
git clone git@github.com:maxspahn/gym_envs_urdf.git
```

Then, you can install the package using pip as:

``` {.sourceCode .bash}
pip3 install .
```

The code can be installed in editible mode using

``` {.sourceCode .bash}
pip3 install -e .
```

Note that we recommend using poetry in this case.

Optional: Installation with poetry
----------------------------------

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

Installing dependencies
-----------------------

Dependencies should be installed through pip or poetry, see below.

Using pip, you can use

``` {.sourceCode .bash}
pip3 install '.[options]'
```

Using poetry

``` {.sourceCode .bash}
poetry install --with <options>
```

Options are `keyboard`.

Examples
--------

You find several python scripts in
[examples/](https://github.com/maxspahn/gym_envs_urdf/tree/master/examples).
You can test those examples using the following (if you use poetry, make
sure to enter the virtual environment first with `poetry shell`)

``` {.sourceCode .python}
python3 pointRobot.py
```

Replace pointRobot.py with the name of the script you want to run.
