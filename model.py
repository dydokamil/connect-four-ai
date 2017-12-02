import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal

from distributions import Categorical


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x,
                                                                        action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x,
                                                                        actions)
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 2, stride=1)
        self.conv3 = nn.Conv2d(64, 32, 2, stride=1)

        self.linear1 = nn.Linear(32 * 2 * 3, 512)

        self.critic_linear = nn.Linear(512, 1)

        num_outputs = action_space.n
        self.dist = Categorical(512, num_outputs)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

    def forward(self, inputs, states, masks):
        # TODO remove `states`
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 2 * 3)
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), x, states
