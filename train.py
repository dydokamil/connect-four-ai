# coding: utf-8

# Credit: https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
import multiprocessing
import os
import threading
from time import sleep

import numpy as np
import tensorflow as tf

from A3CNetwork import A3CNetwork
# ### Helper Functions
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
from Agent import Agent
from EnvironmentManager import EnvironmentManager
from common import s_size, a_size, discount, update_target_graph


# ### Worker Agent
class Worker:
    def __init__(self, game, name, s_size, a_size, trainer, model_path,
                 global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.total_episodes = 0
        self.won = 0

        # Create the local copy of the network and the TensorFlow op
        # to copy global parameters to local network
        self.local_AC = A3CNetwork(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = \
            rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        sess.run([  # self.local_AC.loss,
            # self.local_AC.value_loss,
            # self.local_AC.policy_loss,
            # self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0

                s = self.env.reset()
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                d = False
                while not d:
                    # Take an action using probabilities
                    # from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy,
                         self.local_AC.value,
                         self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    # print(a_dist)
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a == a_dist)

                    s1, r, d = self.env.step(a)

                    r /= 100.

                    if d:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1

                self.total_episodes += 1
                if episode_reward > 0:
                    self.won += 1

                if self.total_episodes % 100 == 0:
                    print(f'Won {self.won}/100')
                    self.won = 0

                self.episode_rewards.append(episode_reward)
                # self.episode_lengths.append(episode_step_count)
                # self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer
                # at the end of the episode.
                if len(episode_buffer) != 0:
                    self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(
                            episode_count) + '.ckpt')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])

                    print("Mean reward:", mean_reward)
                    # mean_length = np.mean(self.episode_lengths[-5:])
                    # mean_value = np.mean(self.episode_mean_values[-5:])

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


if __name__ == '__main__':
    max_episode_length = s_size
    gamma = .99  # discount rate for advantage estimation and reward
    model_yellow = './model_yellow'
    model_red = './model_red'

    tf.reset_default_graph()

    load_model = False
    if not os.path.exists(model_yellow):
        os.makedirs(model_yellow)
    else:
        load_model = True

    if not os.path.exists(model_red):
        os.makedirs(model_red)
    else:
        load_model = True

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32,
                                      name='global_episodes',
                                      trainable=False)
        common_session = tf.Session()
        yellow_session = tf.Session()
        red_session = tf.Session()

        # Generate global network
        with yellow_session:
            trainer_yellow = tf.train.AdamOptimizer('yellow_adam')
            master_network_yellow = A3CNetwork(s_size, a_size,
                                               'global', trainer_yellow)
        with red_session:
            trainer_red = tf.train.AdamOptimizer('red_adam')
            master_network_red = A3CNetwork(s_size, a_size,
                                            'global', trainer_red)

        # Set workers ot number of available CPU threads
        num_workers = multiprocessing.cpu_count()
        assert num_workers % 2 == 0

        managers = []
        with common_session:
            saver = tf.train.Saver(max_to_keep=5)

        for i in range(num_workers):
            num = 0
            agent1 = Agent(num,
                           yellow_session,
                           saver,
                           trainer_yellow)
            agent2 = Agent(num + 1,
                           red_session,
                           saver,
                           trainer_red)
            num += 2
            managers.append(
                EnvironmentManager(agent1, agent2, common_session, coord)
            )

        with common_session:
            common_session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()

            worker_threads = []
            for worker in managers:
                def worker_work(): worker.work(0, gamma, common_session,
                                               coord, saver)

            t = threading.Thread(target=worker_work)
            t.start()
            sleep(.5)
            worker_threads.append(t)
        coord.join(worker_threads)


        # with tf.Session() as sess:
        #     if load_model:
        #         print("Loading saved model.")
        #         saver = tf.train.Saver()
        #         latest_checkpoint = tf.train.latest_checkpoint(model_path)
        #         sess.run(tf.global_variables_initializer())
        #         saver.restore(sess, latest_checkpoint)
        #
        #     coord = tf.train.Coordinator()
        #
        #     # This is where the asynchronous magic happens.
        #     # Start the "work" process for each worker in a separate threat.
        #     worker_threads = []
        #
        #     for worker in workers:
        #         def worker_work(): worker.work(max_episode_length, gamma,
        #                                        sess, coord, saver)
        #
        #
        #         t = threading.Thread(target=worker_work)
        #         t.start()
        #         sleep(0.5)
        #         worker_threads.append(t)
        #     coord.join(worker_threads)
