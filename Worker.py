import numpy as np
import tensorflow as tf

from A3CNetwork import A3CNetwork
from helpers import update_target_graph, discount


class Worker:
    def __init__(self, name, s_size, a_size, trainer, model_path,
                 global_episodes):
        self.name = f'Worker{str(name)}'
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            f'train_{str(self.number)}')

        self.local_A3C = A3CNetwork(s_size, a_size, self.name, trainer)
        self.color = 'yellow' if self.number % 2 == 0 else 'red'
        global_network = \
            'global_yellow' if self.color == 'yellow' else 'global_red'
        self.update_local_ops = update_target_graph(global_network, self.name)

        self.actions = np.identity(a_size, dtype=bool).tolist()
        self.episode_buffer = []
        self.rnn_state = None

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = \
            rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        feed_dict = {self.local_A3C.target_v: discounted_rewards,
                     self.local_A3C.inputs: np.vstack(observations),
                     self.local_A3C.actions: actions,
                     self.local_A3C.advantages: advantages,
                     self.local_A3C.state_in[0]: self.batch_rnn_state[0],
                     self.local_A3C.state_in[1]: self.batch_rnn_state[1]}
        _, _, _, _, _, self.batch_rnn_state, _ = sess.run(
            [self.local_A3C.value_loss,
             self.local_A3C.policy_loss,
             self.local_A3C.entropy,
             self.local_A3C.grad_norms,
             self.local_A3C.var_norms,
             self.local_A3C.state_out,
             self.local_A3C.apply_grads],
            feed_dict=feed_dict)

    def is_it_my_turn(self):
        if self.color == 'yellow':
            return self.env.yellows_turn()

        elif self.color == 'red':
            return not self.env.yellows_turn()

    def action_value_rnn(self, state, sess):
        a_dist, v, rnn_state = sess.run(
            [self.local_A3C.policy,
             self.local_A3C.value,
             self.local_A3C.state_out],
            feed_dict={self.local_A3C.inputs: [state],
                       self.local_A3C.state_in[0]: self.batch_rnn_state[0],
                       self.local_A3C.state_in[1]: self.batch_rnn_state[1]})

        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)  # indexof?
        return self.actions[a], v, rnn_state

    def update_episode_buffer(self, s, a, r, s1, d, v):
        self.episode_buffer.append([s, a, r, s1, d, v[0, 0]])

    def reset_episode_buffer(self):
        self.episode_buffer = []

    def reset_rnn_state(self):
        self.rnn_state = self.local_A3C.state_init
        self.batch_rnn_state = self.local_A3C.state_init
