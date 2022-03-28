from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal

goal1Dict = {
    "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
    'desired_position': [1, 0, 0.1], 'epsilon': 0.02, 'type': "staticSubGoal", 
}

goal1 = StaticSubGoal(name="goal1", contentDict=goal1Dict)
dynamicGoalDict = {
    "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
    'trajectory': ['0.5', '0.2 + 0.2 * ca.sin(0.3 * t)', '0.4'], 'epsilon': 0.08, 'type': "analyticSubGoal", 
}
dynamicGoal = DynamicSubGoal(name="goal2", contentDict=dynamicGoalDict)
splineDict = {'degree': 2, 'controlPoints': [[0.0, -0.0, 0.2], [3.0, 0.0, 2.2], [3.0, 3.0, 1.2]], 'duration': 10}
splineGoalDict = {
    "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
    'trajectory': splineDict, 'epsilon': 0.08, 'type': "splineSubGoal", 
}
splineGoal = DynamicSubGoal(name="goal3", contentDict=splineGoalDict)
