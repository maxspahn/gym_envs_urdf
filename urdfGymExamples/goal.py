from MotionPlanningGoal.staticSubGoal import StaticSubGoal

goal1Dict = {
    "m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
    'desired_position': [1, 0, 0.1], 'epsilon': 0.02, 'type': "staticSubGoal", 
}

goal1 = StaticSubGoal(name="goal1", contentDict=goal1Dict)
