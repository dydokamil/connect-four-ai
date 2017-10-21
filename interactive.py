import numpy as np
import tensorflow as tf

from A3CNetwork import A3CNetwork
from ConnectFourEnvironment import ConnectFourEnvironment
from common import a_size, s_size

env = ConnectFourEnvironment(play_with_rng=False)

tf.reset_default_graph()
model = A3CNetwork(s_size, a_size, 'global', None)

saver = tf.train.Saver()
model_dir = './model'

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, latest_checkpoint)

    while True:
        s = env.reset()
        rnn_state = model.state_init
        d = False
        total_reward = 0
        while not d:
            if env.yellows_turn():
                a_dist, rnn_state = sess.run(
                    [model.policy,
                     model.state_out],
                    feed_dict={model.inputs: [s],
                               model.state_in[0]: rnn_state[0],
                               model.state_in[1]: rnn_state[1]})
                print("Policy:", a_dist)
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a == a_dist)
            else:
                a = input('Choose action (0-6): ')
                a = int(a)

            s, r, d = env.step(a)

            env.render()

            total_reward += r
        print("Reward:", total_reward)
