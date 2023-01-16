from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mpscenes.obstacles.urdf_obstacle import UrdfObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle

import os

obst1Dict = {
    "type": "sphere",
    "geometry": {"position": [2.0, 2.0, 1.0], "radius": 1.0},
}
sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
obst2Dict = {
    "type": "sphere",
    "movable": True,
    "geometry": {"position": [2.0, -0.0, 0.5], "radius": 0.2},
}
sphereObst2 = SphereObstacle(name="simpleSphere", content_dict=obst2Dict)
urdfObst1Dict = {
    "type": "urdf",
    "geometry": {"position": [1.5, 0.0, 0.05]},
    "urdf": os.path.join(os.path.dirname(__file__), "obstacle_data/duck.urdf"),
}
urdfObst1 = UrdfObstacle(name="duckUrdf", content_dict=urdfObst1Dict)
dynamicObst1Dict = {
    "type": "sphere",
    "geometry": {"trajectory": ["2.0 - 0.1 * t", "-0.0", "0.1"], "radius": 0.2},
}
dynamicSphereObst1 = DynamicSphereObstacle(
    name="simpleSphere", content_dict=dynamicObst1Dict
)
dynamicObst2Dict = {
    "type": "analyticSphere",
    "geometry": {"trajectory": ["0.6", "0.5 - 0.1 * t", "0.8"], "radius": 0.2},
}
dynamicSphereObst2 = DynamicSphereObstacle(
    name="simpleSphere", content_dict=dynamicObst2Dict
)
splineDict = {
    "degree": 2,
    "controlPoints": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    "duration": 10,
}
dynamicObst3Dict = {
    "type": "splineSphere",
    "geometry": {"trajectory": splineDict, "radius": 0.2},
}
dynamicSphereObst3 = DynamicSphereObstacle(
    name="simpleSphere", content_dict=dynamicObst3Dict
)

wall_length = 10
wall_obstacles_dicts = [
    {
        'type': 'box', 
         'geometry': {
             'position': [wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': 0.1
        }
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [0.0, wall_length/2.0, 0.4], 'width': 0.1, 'height': 0.8, 'length': wall_length
        }
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [0.0, -wall_length/2.0, 0.4], 'width': 0.1, 'height': 0.8, 'length': wall_length
        }
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [-wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': 0.1
        }
    },
]

wall_obstacles = [BoxObstacle(name=f"wall_{i}", content_dict=obst_dict) for i, obst_dict in enumerate(wall_obstacles_dicts)]
