import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
from highway_env.envs import RacetrackEnv
from typing import Dict, Text
# from utils import record_videos, show_videos

env = gym.make("racetrack-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [5, 5],
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True,
        "steering_range": [-np.pi / 4, np.pi / 4],  # [rad]
        #"acceleration_range": [-2, 2],  # [m/s²]
        "speed_range": [0, 15],  # [m/s]
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}


env.unwrapped.configure(config)
env.reset()
# env = record_videos(env)
# (obs, info), done = env.reset(), False
# for episode in range(100):
#     env.step(2) 

# env.close()
# show_videos()


# show_videos()
# plt.imshow(env.render())
# plt.show()