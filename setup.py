from setuptools import setup

setup(
    name="urdfEnvs",
    version='0.0.1',
    install_requires=['gym',
                      'numpy',
                      'casadi',
                      'pybullet',
                      'urdfpy'],
    extras_require={
        'keyboard_input': ['pynput',
                           'multiprocess']
    }
)
