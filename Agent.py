import numpy as np

from common import discount


class Agent:
    def __init__(self, number, network, optim):
        self.optim = optim
        self.name = f"agent{number}"
        self.network = network
        # self.model = A3CNetwork(s_size, a_size, self.name, trainer)
        # target_graph = 'global_yellow' if number % 2 == 0 else 'global_red'
        # self.update_local_ops = update_target_graph('global', self.name)
        # self.rnn_state = self.model.state_init
        self.epidode_buffer = []

        self.model_path = 'yellow_model' if number % 2 == 0 else 'red_model'

    def add_transition(self, s, a, r, v):
        self.epidode_buffer.append([s, a, r, v])

    def choose_action(self, s):
        a, v, e = self.network(s)

        return a, v, e

    # def save_model(self, episode_count):
    #     self.saver.save(self.session,
    #                     self.model_path + '/model-' + episode_count)

    def train(self, gamma):
        rollout = np.array(self.epidode_buffer)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = np.asarray(rollout[:, 3])

        discounted_rewards = discount(rewards, gamma).astype(np.float32)
        advantages = rewards + gamma * values - values  # why not discounted ??
        advantages = discount(advantages, gamma)

        loss = self.network.loss(discounted_rewards, actions, advantages)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
