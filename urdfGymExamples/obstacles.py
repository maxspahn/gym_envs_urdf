from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle

import os

obst1Dict = {
    "dim": 3,
    "type": "sphere",
    "geometry": {"position": [2.0, 2.0, 1.0], "radius": 1.0},
}
sphereObst1 = SphereObstacle(name="simpleSphere", contentDict=obst1Dict)
obst2Dict = {
    "dim": 3,
    "type": "sphere",
    'movable': True,
    "geometry": {"position": [2.0, -0.0, 0.5], "radius": 0.2},
}
sphereObst2 = SphereObstacle(name="simpleSphere", contentDict=obst2Dict)
urdfObst1Dict = {
    'dim': 3,
    'type': 'urdf',
    'geometry': {'position': [1.5, 0.0, 0.05]},
    'urdf': os.path.join(os.path.dirname(__file__), 'obstacleData/duck.urdf'),
}
urdfObst1 = UrdfObstacle(name='duckUrdf', contentDict=urdfObst1Dict)
dynamicObst1Dict = {
    "dim": 3,
    "type": "sphere",
    "geometry": {"trajectory": ['2.0 - 0.1 * t', '-0.0', '0.1'], "radius": 0.2},
}
dynamicSphereObst1 = DynamicSphereObstacle(name="simpleSphere", contentDict=dynamicObst1Dict)
