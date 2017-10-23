import copy
import os

import numpy as np
import torch

from common import discount


class Agent:
    def __init__(self, number, network, optim):
        self.optim = optim
        self.name = f"agent{number}"
        self.network_original = network
        self.network = copy.deepcopy(self.network_original)
        self.epidode_buffer = []
        self.losses = []

        self.model_path = 'yellow_model' if number % 2 == 0 else 'red_model'

    def add_transition(self, a, r, v):
        self.epidode_buffer.append([a, r, v])

    def log_loss(self, loss):
        self.losses.append(loss)
        if len(self.losses) == 100:
            mean = np.mean(self.losses)
            self.losses = []
            return mean

    def reset_agent(self):
        self.epidode_buffer = []
        self.network = copy.deepcopy(self.network_original)

    def choose_action(self, s):
        a, v, e = self.network(s)

        return a, v, e

    # def save_model(self, episode_count):
    #     self.saver.save(self.session,
    #                     self.model_path + '/model-' + episode_count)

    def train(self, gamma):
        rollout = np.array(self.epidode_buffer)
        actions = rollout[:, 0]
        rewards = rollout[:, 1]
        values = np.asarray(rollout[:, 2])

        discounted_rewards = discount(rewards, gamma).astype(np.float32)
        advantages = rewards + gamma * values - values  # why not discounted ??
        advantages = discount(advantages, gamma)

        loss = self.network.compute_loss(discounted_rewards,
                                         actions,
                                         advantages)

        loss_mean = self.log_loss(loss.data[0])
        if loss_mean is not None:
            print(loss_mean)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.network_original.load_state_dict(self.network.state_dict())
        # self.network_original = copy.deepcopy(self.network)

    def save(self):
        if self.name == 'agent0' or self.name == 'agent1':
            torch.save(self.network, os.path.join(self.model_path,
                                                  self.model_path))
