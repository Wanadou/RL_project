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


class ReplayBuffer:
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


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQN:
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
        hidden_size,
    ):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        self.reset()

    def update(self, state, action, reward, terminated, next_state):
        """
        ** SOLUTION **
        """

        # add data to replay buffer
        self.buffer.push(
            torch.tensor(state.reshape(-1)).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state.reshape(-1)).unsqueeze(0),
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

        # print(f"State batch ({state_batch.shape}): {state_batch}")
        # print(f"Action batch ({action_batch.shape}): {action_batch}")


        values = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch


        targets = targets.unsqueeze(1).float()
        # print(f"Values ({values.shape}, {values.dtype}): {values}")
        # print(f"Targets ({targets.shape}, {targets.dtype}): {targets}")
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

        return loss.detach().numpy()

    def get_q(self, state):
        """
        Compute Q function for a states
        """
        # reshape state to 1 dimension
        reshaped_state = state.reshape(-1)
        # print(f"State: {reshaped_state}")
        # print(f"State shape: {reshaped_state.shape}")
        state_tensor = torch.tensor(reshaped_state).unsqueeze(0)
        with torch.no_grad():
            # print(f"State tensor: {state_tensor}")
            # print(f"State tensor shape: {state_tensor.shape}")
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)

    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            # print("Random action")
            return self.env.action_space.sample()
        else:
            # print("Greedy action")
            return np.argmax(self.get_q(state))

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )

    def reset(self):

        obs_size = self.observation_space.shape[0] * self.observation_space.shape[1]
        n_actions = self.action_space.n
        # print(f"Observation size: {obs_size}")
        # print(f"Action size: {n_actions}")

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_size, self.hidden_size, n_actions)
        self.target_net = Net(obs_size, self.hidden_size, n_actions)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
