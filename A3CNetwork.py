import numpy as np
import tensorflow as tf


def weight_variable(name, shape):
    return tf.get_variable(name,
                           shape=shape,
                           initializer=tf.glorot_uniform_initializer())


class A3CNetwork:
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, s_size],
                                         dtype=tf.float32, name='inputs')
            self.inputs_rect = tf.reshape(self.inputs, shape=[-1, 6, 7, 1])
            # self.imageIn = tf.reshape(self.inputs, shape=[-1, 6, 7])

            W_conv1 = weight_variable('W_conv1', [2, 2, 1, 64])
            b_conv1 = weight_variable('b_conv1', [64])
            h_conv1 = tf.nn.conv2d(self.inputs_rect,
                                   W_conv1,
                                   [1, 1, 1, 1],
                                   padding='VALID') + b_conv1

            W_conv2 = weight_variable('W_conv2', [2, 2, 64, 64])
            b_conv2 = weight_variable('b_conv2', [64])
            h_conv2 = tf.nn.conv2d(h_conv1,
                                   W_conv2,
                                   [1, 1, 1, 1],
                                   padding='VALID') + b_conv2

            h_conv2_flattened = tf.reshape(h_conv2, [-1, 4 * 5 * 64])

            W_h1 = weight_variable('W_h1', [4 * 5 * 64, 200])
            b_h1 = weight_variable('b_h1', [200])
            hidden1 = tf.nn.elu(tf.matmul(h_conv2_flattened, W_h1) + b_h1)

            W_h2 = weight_variable('W_h2', [200, 128])
            b_h2 = weight_variable('b_h2', [128])
            hidden2 = tf.nn.elu(tf.matmul(hidden1, W_h2) + b_h2)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c],
                                  name='c_in')
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h],
                                  name='h_in')
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden2, [0])
            step_size = tf.shape(self.inputs_rect)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in,
                sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 128])

            W_policy = weight_variable('W_policy', [128, a_size])
            self.policy = tf.nn.softmax(tf.matmul(rnn_out, W_policy),
                                        name='policy')

            W_value = weight_variable('W_value', [128, 1])
            self.value = tf.matmul(rnn_out, W_value)

            # Only the worker network need ops for loss functions
            # and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size,
                                                 dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],
                                                 dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(
                    self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value, [-1])))

                self.entropy = - tf.reduce_sum(
                    self.policy * tf.log(self.policy))

                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs) * self.advantages)

                self.loss = (0.5
                             * self.value_loss
                             + self.policy_loss
                             - self.entropy
                             * 0.01)

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = \
                    tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))
