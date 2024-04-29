import gymnasium as gym
import numpy as np

env = gym.make("highway-fast-v0", render_mode="rgb_array")

# config = {
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": 3,
#         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "heading", "cos_d", "sin_d", "long_off", "lat_off", "ang_off"],
#         "features_range": {
#             "x": [-100, 100],
#             "y": [-100, 100],
#             "vx": [-20, 20],
#             "vy": [-20, 20],
#         },
#         "grid_size": [[-20, 20], [-20, 20]],
#         "grid_step": [5, 5],
#         "absolute": False,
#     },
#     "action": {
#         "type": "DiscreteAction",
#         "longitudinal": True,
#         # "acceleration_range" : (0, 10),  # [m/s²]
#         "speed_range": [0, 30],  # [m/s]
#         "lateral": True,
#         "steering_range": [-np.pi / 4, np.pi / 4],
#         "actions_per_axis": 3,

#     },
#     "lanes_count": 2,
#     "vehicles_count": 3,
#     "duration": 20,  # [s]
#     "initial_spacing": 0,
#     "collision_reward": -1,  # The reward received when colliding with a vehicle.
#     "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
#     # zero for other lanes.
#     "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
#     # lower speeds according to config["reward_speed_range"].
#     "lane_change_reward": 0,
#     "reward_speed_range": [
#         20,
#         30,
#     ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
#     "normalize_reward": True,
#     "simulation_frequency": 15,  # [Hz]
#     "policy_frequency": 1,  # [Hz]
#     "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
#     "screen_width": 600,  # [px]
#     "screen_height": 150,  # [px]
#     "centering_position": [0.3, 0.5],
#     "scaling": 5.5,
#     "show_trajectories": True,
#     "render_agent": True,
#     "offscreen_rendering": False,
#     "disable_collision_checks": True,
# }

# config = {
#     "action": {"type": "DiscreteMetaAction"},
#     "centering_position": [0.3, 0.5],
#     "collision_reward": -1,
#     "controlled_vehicles": 1,
#     "duration": 30,
#     "ego_spacing": 1.5,
#     "high_speed_reward": 0.4,
#     "initial_lane_id": None,
#     "lane_change_reward": 0,
#     "lanes_count": 3,
#     "manual_control": False,
#     "observation": {"type": "Kinematics"},
#     "offroad_terminal": False,
#     "offscreen_rendering": False,
#     "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
#     "policy_frequency": 1,
#     "real_time_rendering": False,
#     "render_agent": True,
#     "reward_speed_range": [20, 30],
#     "right_lane_reward": 0.1,
#     "scaling": 5.5,
#     "screen_height": 150,
#     "screen_width": 600,
#     "show_trajectories": False,
#     "simulation_frequency": 5,
#     "vehicles_count": 20,
#     "vehicles_density": 1,
# }


# env.unwrapped.configure(config)
# env.reset()


# import gymnasium as gym
# import highway_env
# import numpy as np
# import matplotlib.pyplot as plt
# from utils import record_videos, show_videos


config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": True,
    },
    "action": {
        "type": "DiscreteAction",
        "longitudinal": True,  # enable throttle control
        "lateral": True,  # enable steering control
        "dynamical": False,  # whether to simulate dynamics (i.e. friction) rather than kinematics
        "steering_range": [-np.pi / 8, np.pi / 8],  # [rad]
        "acceleration_range": [-2, 2],  # [m/s²]
        "speed_range": [0, 15],  # [m/s]
        "actions_per_axis": 3,
    },
    "lanes_count": 4,
    "vehicles_count": 10,
    "duration": 40,  # [s]
    "initial_spacing": 0,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
    "high_speed_reward": 0.5,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0.2,
    "reward_speed_range": [
        5,
        15,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
    "normalize_reward": True,
}
"""


config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "DiscreteAction",
    },
    "lanes_count": 3,
    "vehicles_count": 10,
    "duration": 20,  # [s]
    "initial_spacing": 0,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.1,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False, #True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}
"""

env.unwrapped.configure(config)
env.reset()

# env = record_videos(env)
# (obs, info), done = env.reset(), False
# print(env.action_space)
# for episode in range(100):
#     action = env.action_space.sample()
#     print(f'Action: {action}')
#     env.step(action)

# env.close()
# show_videos()


# show_videos()
# plt.imshow(env.render())
# plt.show()
