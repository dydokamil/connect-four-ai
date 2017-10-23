import numpy as np

from common import discount


class Agent:
    def __init__(self, number, network, optim):
        self.optim = optim
        self.name = f"agent{number}"
        self.network = network
        self.epidode_buffer = []
        self.losses = []

        self.model_path = 'yellow_model' if number % 2 == 0 else 'red_model'

    def add_transition(self, s, a, r, v):
        self.epidode_buffer.append([s, a, r, v])

    def log_loss(self, loss):
        self.losses.append(loss)
        if len(self.losses) == 100:
            mean = np.mean(self.losses)
            self.losses = []
            return mean

    def clear_buffer(self):
        self.epidode_buffer = []

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

        loss = self.network.compute_loss(discounted_rewards,
                                         actions,
                                         advantages)

        loss_mean = self.log_loss(loss.data[0])
        if loss_mean is not None:
            print(loss_mean)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
