import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from common import s_size, a_size, one_hot_encode


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_policy = nn.Linear(512, a_size)
        self.fc_value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        self.policy = F.softmax(self.fc_policy(x))
        self.value = F.elu(self.fc_value(x))
        self.entropy = torch.sum(self.policy * torch.log(self.policy))

        return self.policy, self.value, self.entropy

    def loss(self, discounted_rewards, actions, advantages):
        self.actions_oh = Variable(torch.from_numpy(
            one_hot_encode(actions, a_size).astype(np.float32))).t()
        self.responsible_outputs = torch.sum(torch.matmul(self.policy,
                                                          self.actions_oh))

        advantages = advantages[0]

        discounted_rewards = Variable(torch.from_numpy(discounted_rewards))
        self.value_loss = (.5 * torch.sum((discounted_rewards
                                           - self.value) ** 2))
        self.policy_loss = -torch.sum(
            torch.matmul(torch.log(self.responsible_outputs), advantages))
        self.loss = (.5
                     * self.value_loss
                     + self.policy_loss
                     - self.entropy
                     * 1e-4)
        return self.loss
