import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from utils import record_videos, show_videos

env = gym.make("parking-v0", render_mode="rgb_array")

config = {
    {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction"
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
    }
}


env.unwrapped.configure(config)

env = record_videos(env)
(obs, info), done = env.reset(), False
for episode in range(100):
    env.step(2) 

env.close()
show_videos()


# show_videos()
# plt.imshow(env.render())
# plt.show()