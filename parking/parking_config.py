import gymnasium as gym
import highway_env

# import matplotlib.pyplot as plt
# from utils import record_videos, show_videos

env = gym.make("parking-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}


env.unwrapped.configure(config)
env.reset()


# BASE CONFIG

# {'action': {'type': 'ContinuousAction'},
# 'add_walls': True,
# 'centering_position': [0.5, 0.5],
# 'collision_reward': -5,
# 'controlled_vehicles': 1,
# 'duration': 100,
# 'manual_control': False,
# 'observation': {'features': ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
#                 'normalize': False,
#                 'scales': [100, 100, 5, 5, 1, 1],
#                 'type': 'KinematicsGoal'},
# 'offscreen_rendering': False,
# 'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
# 'policy_frequency': 5,
# 'real_time_rendering': False,
# 'render_agent': True,
# 'reward_weights': [1, 0.3, 0, 0, 0.02, 0.02],
# 'scaling': 7,
# 'screen_height': 300,
# 'screen_width': 600,
# 'show_trajectories': False,
# 'simulation_frequency': 15,
# 'steering_range': 0.7853981633974483,
# 'success_goal_reward': 0.12,
# 'vehicles_count': 0}

# env = record_videos(env)
# (obs, info), done = env.reset(), False
# for episode in range(100):
#     env.step(2)

# env.close()
# show_videos()


# show_videos()
# plt.imshow(env.render())
# plt.show()
