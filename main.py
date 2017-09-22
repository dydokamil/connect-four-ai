import multiprocessing
import os
import threading
from time import sleep

import tensorflow as tf

from A3CNetwork import A3CNetwork
from ConnectFourEnvironment import ConnectFourEnvironment
from Worker import Worker

max_episode_length = 6 * 7
gamma = .99
s_size = 6 * 7
a_size = 7
load_model = False
model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device('/cpu:0'):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',
                                  trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

    master_network = A3CNetwork(s_size, a_size, 'global', None)
    num_workers = multiprocessing.cpu_count()
    workers = []
    for i in range(num_workers):
        workers.append(Worker(ConnectFourEnvironment(), i, s_size, a_size,
                              trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print("Loading model...")
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,
                                          gamma,
                                          sess,
                                          coord,
                                          saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(.5)
        worker_threads.append(t)
    coord.join(worker_threads)
