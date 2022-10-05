from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle

import os

obst1Dict = {
    "type": "sphere",
    "geometry": {"position": [2.0, 2.0, 1.0], "radius": 1.0},
}
sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
obst2Dict = {
    "type": "sphere",
    'movable': True,
    "geometry": {"position": [2.0, -0.0, 0.5], "radius": 0.2},
}
sphereObst2 = SphereObstacle(name="simpleSphere", content_dict=obst2Dict)
urdfObst1Dict = {
    'type': 'urdf',
    'geometry': {'position': [1.5, 0.0, 0.05]},
    'urdf': os.path.join(os.path.dirname(__file__), 'obstacle_data/duck.urdf'),
}
urdfObst1 = UrdfObstacle(name='duckUrdf', content_dict=urdfObst1Dict)
dynamicObst1Dict = {
    "type": "sphere",
    "geometry": {"trajectory": ['2.0 - 0.1 * t', '-0.0', '0.1'], "radius": 0.2},
}
dynamicSphereObst1 = DynamicSphereObstacle(name="simpleSphere", content_dict=dynamicObst1Dict)
dynamicObst2Dict = {
    "type": "analyticSphere",
    "geometry": {"trajectory": ['0.6', '0.5 - 0.1 * t', '0.8'], "radius": 0.2},
}
dynamicSphereObst2 = DynamicSphereObstacle(name="simpleSphere", content_dict=dynamicObst2Dict)
splineDict = {'degree': 2, 'controlPoints': [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], 'duration': 10}
dynamicObst3Dict = {
    "type": "splineSphere",
    "geometry": {"trajectory": splineDict, "radius": 0.2},
}
dynamicSphereObst3 = DynamicSphereObstacle(name="simpleSphere", content_dict=dynamicObst3Dict)
