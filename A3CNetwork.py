import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


class A3CNetwork:
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, s_size],
                                         dtype=tf.float32)
            self.image = tf.reshape(self.inputs, shape=[-1, 6, 7, 1])
            self.conv1 = slim.conv2d(self.image, num_outputs=16,
                                     kernel_size=[3, 3],
                                     stride=[1, 1],
                                     padding='SAME',
                                     activation_fn=tf.nn.elu)
            self.conv2 = slim.conv2d(self.conv1,
                                     num_outputs=32,
                                     kernel_size=[3, 3],
                                     stride=[1, 1],
                                     padding='SAME',
                                     activation_fn=tf.nn.elu)
            hidden = slim.fully_connected(slim.flatten(self.conv2),
                                          256,
                                          activation_fn=tf.nn.elu)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.image)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in,
                sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = lstm_c[:1, :], lstm_h[:1, :]
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # policy pi
            self.policy = slim.fully_connected(rnn_out,
                                               num_outputs=a_size,
                                               activation_fn=tf.nn.softmax)
            # value V
            self.value = slim.fully_connected(rnn_out,
                                              num_outputs=1,
                                              activation_fn=None)

            # if worker then functions for gradient updating
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],
                                              dtype=tf.int32)
                self.actions_oh = tf.one_hot(self.actions, a_size,
                                             dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],
                                               dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],
                                                 dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(
                    self.policy * self.actions_oh, [1])

                # loss
                self.value_loss = .5 * tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(
                    self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs) * self.advantages)
                self.loss = (self.value_loss
                             * .5
                             + self.policy_loss
                             - self.entropy
                             * .01)
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = \
                    tf.clip_by_global_norm(self.gradients, 40.)
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = \
                    trainer.apply_gradients(zip(grads, global_vars))
