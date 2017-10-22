import numpy as np

from A3CNetwork import A3CNetwork
from common import discount, s_size, a_size, update_target_graph


class Agent:
    def __init__(self, number, session, saver, trainer):
        self.name = f"agent{number}"
        self.session = session
        self.saver = saver
        self.trainer = trainer
        self.model = A3CNetwork(s_size, a_size, self.name, trainer)
        target_graph = 'global_yellow' if number % 2 == 0 else 'global_red'
        self.update_local_ops = update_target_graph('global', self.name)
        self.rnn_state = self.model.state_init
        self.epidode_buffer = []

        self.model_path = 'yellow_model' if number % 2 == 0 else 'red_model'

    def get_rnn_state(self):
        return self.rnn_state

    def add_transition(self, s, a, r, s1, d, v):
        self.epidode_buffer.append([s, a, r, s1, d, v])

    def choose_action(self, s):
        a_dist, v, rnn_state = self.session.run(
            [self.model.policy,
             self.model.value,
             self.model.state_out],
            feed_dict={self.model.local_AC.inputs: [s],
                       self.model.state_in[0]: self.rnn_state[0],
                       self.model.state_in[1]: self.rnn_state[1]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a == a_dist)

        self.rnn_state = rnn_state

        return a, v

    def save_model(self, episode_count):
        self.saver.save(self.session,
                        self.model_path + '/model-' + episode_count)

    def reset_agent(self):
        self.session.run(self.model.update_local_ops)
        self.rnn_state = self.model.state_init

    def train(self, gamma):
        rollout = np.array(self.epidode_buffer)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = np.asarray(rollout[:, 3])

        discounted_rewards = discount(rewards, gamma)
        advantages = rewards + gamma * values - values  # why not discounted ??
        advantages = discount(advantages, gamma)

        feed_dict = {self.model.target_v: discounted_rewards,
                     self.model.inputs: np.vstack(observations),
                     self.model.actions: actions,
                     self.model.advantages: advantages,
                     # self.model.state_in[0]: self.rnn_state[0],
                     # self.model.state_in[1]: self.rnn_state[1],
                     }
        self.session.run([self.model.grad_norms,
                          self.model.var_norms,
                          self.model.state_out,
                          self.model.apply_grads],
                         feed_dict=feed_dict)
