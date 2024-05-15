import numpy as np
import gymnasium as gym

from robotmodels.utils.robotmodel import RobotModel
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from urdfenvs.scene_examples.goal import goal1
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot


obstacle_configuration = {
    "type": "sphere",
    "geometry": {"position": [2.0, 2.0, 1.0], "radius": 0.3},
    "rgba": [0.3, 0.5, 0.6, 1.0],
    'low': {
        'position' : [2.0, 2.0, 0.1],
         'radius': 0.1,
    },
    'high': {
         'position' : [5.0, 5.0, 0.3],
         'radius': 0.3,
    },
}
obstacle = SphereObstacle(name="sphere_1", content_dict=obstacle_configuration)

def test_reset_mujoco():
    """
    Test if the obstacle position changes between two runs of the same environment.
    """
    options = {
        "randomize_obstacles": True,
        "randomize_goals": True,
    }
    sensor = FullSensor(
        goal_mask=['position'],
        obstacle_mask=["position", "size"],
        variance=0.0,
        physics_engine_name="mujoco",
    )
    robot_model = RobotModel("pointRobot", "pointRobot")

    xml_file = robot_model.get_xml_path()
    robots = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    env = gym.make(
        "generic-mujoco-env-v0",
        robots=robots,
        obstacles=[obstacle],
        goals=[goal1],
        sensors=[sensor],
        render=False,
        enforce_real_time=False,
    ).unwrapped
    env.reset()
    action = np.random.random(env.nu)
    number_of_steps = 10

    obstacle_position_run_0 = np.zeros(3)
    obstacle_position_run_1 = np.zeros(3)
    goal_position_run_0 = np.zeros(3)
    goal_position_run_1 = np.zeros(3)

    for _ in range(number_of_steps):
        ob, *_ = env.step(action)
        observation_run_0 = ob['robot_0']['joint_state']['position']
        obstacle_position_run_0 = ob['robot_0']['FullSensor']['obstacles'][0]['position']
        goal_position_run_0 = ob['robot_0']['FullSensor']['goals'][0]['position']

    assert np.allclose(obstacle_position_run_0, obstacle.position())
    assert np.allclose(goal_position_run_0, goal1.position())

    env.reset(options=options)
    for _ in range(number_of_steps):
        ob, *_ = env.step(action)
        observation_run_1 = ob['robot_0']['joint_state']['position']
        obstacle_position_run_1 = ob['robot_0']['FullSensor']['obstacles'][0]['position']
        goal_position_run_1 = ob['robot_0']['FullSensor']['goals'][0]['position']

    assert np.allclose(observation_run_0, observation_run_1)
    assert not np.allclose(obstacle_position_run_0, obstacle_position_run_1)
    assert not np.allclose(goal_position_run_0, goal_position_run_1)
