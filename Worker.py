import numpy as np
import tensorflow as tf

from A3CNetwork import A3CNetwork
from helpers import update_target_graph, discount


class Worker:
    def __init__(self, environment, name, s_size, a_size, trainer, model_path,
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
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = environment

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
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
        sess.run([self.local_A3C.value_loss,
                  self.local_A3C.policy_loss,
                  self.local_A3C.entropy,
                  self.local_A3C.grad_norms,
                  self.local_A3C.var_norms,
                  self.local_A3C.state_out,
                  self.local_A3C.apply_grads],
                 feed_dict=feed_dict)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        print(f"Starting worker {self.number}")
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                s = self.env.reset().flatten()
                rnn_state = self.local_A3C.state_init
                self.batch_rnn_state = rnn_state
                while not self.env.is_finished():
                    a_dist, v, rnn_state = sess.run(
                        [self.local_A3C.policy,
                         self.local_A3C.value,
                         self.local_A3C.state_out],
                        feed_dict={self.local_A3C.inputs: [s],
                                   self.local_A3C.state_in[0]: rnn_state[0],
                                   self.local_A3C.state_in[1]: rnn_state[1]})

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)  # indexof?

                    s1, r, d = self.env.step(self.actions[a])
                    if not d:
                        s1 = self.env.get_state().flatten()
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    episode_step_count += 1

                    if len(episode_buffer) == 5 \
                            and not d \
                            and episode_step_count != max_episode_length - 1:
                        v1 = sess.run(self.local_A3C.value,
                                      feed_dict={self.local_A3C.inputs: [s],
                                                 self.local_A3C.state_in[0]:
                                                     rnn_state[0],
                                                 self.local_A3C.state_in[1]:
                                                     rnn_state[1]})[0, 0]
                        self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    elif d:
                        break

                self.episode_rewards.append(episode_step_count)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    self.train(episode_buffer, sess, gamma, .0)

                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'Worker0':
                        saver.save(sess,
                                   f'{self.model_path}/model-{str(episode_count)}.cptk)')
                        print("Saved model")

                if self.name == 'Worker0':
                    sess.run(self.increment)
                episode_count += 1
