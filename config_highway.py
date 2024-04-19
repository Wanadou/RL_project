import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from utils import record_videos, show_videos

env = gym.make("highway-fast-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 3,
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
        "longitudinal": True, # enable throttle control
        "lateral": False, # enable steering control
        "dynamical": False, # whether to simulate dynamics (i.e. friction) rather than kinematics
        "steering_range": [-np.pi / 4, np.pi / 4], # [rad]
        "acceleration_range": [-5, 5], # [m/s²] 
        "speed_range": [0, 10] # [m/s]
    },
    "lanes_count": 4,
    "vehicles_count": 10,
    "duration": 40,  # [s]
    "initial_spacing": 0,
    "collision_reward": -10,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
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
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}

env.unwrapped.configure(config)

env = record_videos(env)
(obs, info), done = env.reset(), False
print(env.action_space)
for episode in range(100):
    action = env.action_space.sample()
    print(f'Action: {action}')
    env.step(action) 

env.close()
show_videos()


# show_videos()
# plt.imshow(env.render())
# plt.show()