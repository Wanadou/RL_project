# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy

# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# from IPython.display import clear_output

import matplotlib.pyplot as plt

class ReplayBuffer_v2:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)


class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=2, stride=1, padding=1),  # Outputs 9x9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size to 4x4
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),  # Reduces size to 3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # Reduces size to 2x2 (optional additional pooling)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2 , 64),  # Adjusted to match the output from the last pooling layer
            nn.ReLU(),
            nn.Linear(64, output_size)  # Output 2 values (throttle and steering angle)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class DQN_v2:
    def __init__(
        self,
        env,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        input_channels,
    ):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every
        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.buffer = ReplayBuffer_v2(self.buffer_capacity)
        self.q_net = CNN(7, 2)  # Output 2 values (throttle and steering angle)
        self.target_net = CNN(7, 2)  # Output 2 values (throttle and steering angle)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)
        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
        self.reset()
    
    def update(self, state, action, reward, terminated, next_state):
        # add data to replay buffer
        self.buffer.push(
            torch.tensor(state).float().unsqueeze(0),
            torch.tensor(action).float().unsqueeze(0),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state).float().unsqueeze(0),
        )
        if len(self.buffer) < self.batch_size:
            return np.inf
        # get batch
        transitions = self.buffer.sample(self.batch_size)
        (
            state_batch,
            action_batch,
            reward_batch,
            terminated_batch,
            next_state_batch,
        ) = tuple([torch.cat(data) for data in zip(*transitions)])
        values = self.q_net.forward(state_batch)
        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch
        #######################
        # force targets dtype fo Double (float32)
        # targets = targets.unsqueeze(1).float()
        #######################
        loss = self.loss_function(values, targets)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()
        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.decrease_epsilon()
        self.n_steps += 1
        if terminated:
            self.n_eps += 1
        # return loss.detach().numpy()
        return loss.item()
    
    def get_q(self, state):
        """
        Compute Q function for a states
        """
        #######################
        # reshape state to 1 dimension
        # reshaped_state = state.reshape(-1)
        state_tensor = torch.tensor(state).unsqueeze(0)
        #######################
        with torch.no_grad():
            output = self.q_net.forward(state_tensor)  # shape (1, 2)
        return output.numpy()[0]  # shape (2)
    
    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            # print("Random action")
            return np.array([random.uniform(-2, 2), random.uniform(-np.pi/4, np.pi/4)])
        else:
            # print("Greedy action")
            return self.get_q(state)
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )
    
    def reset(self):
        #######################
        # compute total number of observations
        # (nedd for np.product() since the shape of the observation_space tensor can vary depending on the configured observations)
        obs_size = np.product(self.observation_space.shape)
        #######################
        self.buffer = ReplayBuffer_v2(self.buffer_capacity)
        self.q_net = CNN(self.input_channels, 2)  # Output 2 values (throttle and steering angle)
        self.target_net = CNN(self.input_channels, 2)  # Output 2 values (throttle and steering angle)
        #######################
        print(f"Q CNN: {self.q_net}")
        #######################
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )
        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
    
    def save_state(self, filename='dqn_model.pth'):
        """
        Save the model state to a file.
        """
        torch.save(self.q_net.state_dict(), filename)
        print(f"Model saved to {filename}")
    
    def load_state(self, filename='dqn_model.pth'):
        """
        Load the model state from a file.
        """
        self.q_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
        self.target_net.eval()
        print(f"Model loaded from {filename}")
