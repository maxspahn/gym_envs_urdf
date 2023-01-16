from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.goals.dynamic_sub_goal import DynamicSubGoal

goal1Dict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [0, 1, 2],
    "parent_link": 0,
    "child_link": 3,
    "desired_position": [1, 0, 0.1],
    "epsilon": 0.02,
    "type": "staticSubGoal",
}

goal1 = StaticSubGoal(name="goal1", content_dict=goal1Dict)
dynamicGoalDict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [0, 1, 2],
    "parent_link": 0,
    "child_link": 3,
    "trajectory": ["0.5", "0.2 + 0.2 * sp.sin(0.3 * t)", "0.4"],
    "epsilon": 0.08,
    "type": "analyticSubGoal",
}
dynamicGoal = DynamicSubGoal(name="goal2", content_dict=dynamicGoalDict)
splineDict = {
    "degree": 2,
    "controlPoints": [[0.0, -0.0, 0.2], [3.0, 0.0, 2.2], [3.0, 3.0, 1.2]],
    "duration": 10,
}
splineGoalDict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [0, 1, 2],
    "parent_link": 0,
    "child_link": 3,
    "trajectory": splineDict,
    "epsilon": 0.08,
    "type": "splineSubGoal",
}
splineGoal = DynamicSubGoal(name="goal3", content_dict=splineGoalDict)
