# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy

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
    def __init__(self, input_channels, width, height):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        # Calculate the output dimensions after the convolutions and pooling
        # out_width = width // 8
        # out_height = height // 8

        out_width = 4
        out_height = 4
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * out_width * out_height, 16),  
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



class REINFORCE:
    def __init__(
        self,
        env,
        gamma,
        batch_size,
        buffer_capacity,
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

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.input_channels = input_channels

        self.buffer = ReplayBuffer_v2(self.buffer_capacity)
        self.policy_net = CNN(input_channels = 2, width = 35, height = 22)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
        self.reset()

    def update(self, state, action, reward, terminated, next_state):
        """
        ** SOLUTION **
        """


        self.current_episode.append((
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
        )
        )

        if terminated:


            # add data to replay buffer
            self.buffer.push(
                torch.tensor(state).float().unsqueeze(0),
                torch.tensor([[action]], dtype=torch.int64),
                torch.tensor([reward]),
                torch.tensor([terminated], dtype=torch.int64),
                torch.tensor(next_state).float().unsqueeze(0),
            )

            if len(self.buffer) < self.batch_size:
                return np.inf

            # get batch
            transitions = self.buffer.sample(self.batch_size)

            self.n_eps += 1

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            current_episode_returns = self._gradient_returns(rewards, self.gamma)
            current_episode_returns = (current_episode_returns - current_episode_returns.mean())

            # Assuming self.policy_net.forward(states) returns a tuple (mean, std_dev)
            mean_std_dev = self.policy_net.forward(states)

            # Construct a normal distribution object
            normal_dist = torch.distributions.Normal(mean_std_dev[0][0], mean_std_dev[0][1])

            # Log probabilities for the actions of the episode
            log_probs = normal_dist.log_prob(actions)

            self.scores.append(torch.dot(log_probs.squeeze(), current_episode_returns).unsqueeze(0))
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size)==0:
                self.optimizer.zero_grad()
                full_neg_score = - torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()
                self.optimizer.step()
                
                self.scores = []

            self.decrease_epsilon()

            self.n_steps += 1
        
    
    def _gradient_returns(self, rewards, gamma):
        """
        Turns a list of rewards into the list of returns * gamma**t
        """
        G = 0
        returns_list = []
        T = len(rewards)
        full_gamma = np.power(gamma, T)
        for t in range(T):
            G = rewards[T-t-1] + gamma * G
            full_gamma /= gamma
            returns_list.append(full_gamma * G)
        return torch.tensor(returns_list[::-1])#, dtype=torch.float32)

    def get_action(self, state, epsilon=None):
        

        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():

            #print(state_tensor)
            mean_variance = self.policy_net.forward(state_tensor).numpy()
            #print(mean_variance)
            # p = np.exp(unn_log_probs - np.min(unn_log_probs))
            # p = p /  np.sum(p)
            return (np.random.normal(mean_variance[0][0], np.sqrt(np.exp(mean_variance[0][1]))), epsilon)
            # return np.random.choice(np.arange(self.action_space.n), p=p)
            


    def reset(self):
        hidden_size = 128

        self.policy_net = CNN(input_channels = self.input_channels, width = 35, height = 22)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0



    def save_state(self, filename='dqn_model.pth'):
        """
        Save the model state to a file.
        """
        torch.save(self.q_CNN.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_state(self, filename='dqn_model.pth'):
        """
        Load the model state from a file.
        """
        self.q_CNN.load_state_dict(torch.load(filename))
        self.target_CNN.load_state_dict(self.q_CNN.state_dict())
        self.q_CNN.eval()
        self.target_CNN.eval()
        print(f"Model loaded from {filename}")