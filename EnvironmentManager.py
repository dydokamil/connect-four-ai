class EnvironmentManager:
    def __init__(self, environment, player1, player2):
        self.env = environment
        self.player1 = player1
        self.player2 = player2

    def get_action_value_rnn(self, state, sess):
        if self.env.yellows_turn():
            return self.player1.action_value_rnn(state, sess)
        else:
            return self.player2.action_value_rnn(state, sess)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.player1.update_local_ops)
                sess.run(self.player2.update_local_ops)

                self.player1.reset_rnn_state()
                self.player2.reset_rnn_state()

                self.player1.reset_episode_buffer()
                self.player2.reset_episode_buffer()

                if self.env.is_finished():
                    s = self.env.reset()

                while not self.env.is_finished():
                    is_p1_turn = self.env.yellows_turn()
                    s = self.env.get_state().flatten()
                    a, v, rnn_state = self.get_action_value_rnn(s, sess)
                    self.env.step(a)
                    d = self.env.is_finished()
                    r = self.env.get_reward()
                    if d:
                        s1 = s
                    else:
                        s1 = self.env.get_state().flatten()

                    if is_p1_turn:
                        self.player1.update_episode_buffer(s, a, r, s1, d, v)
                    else:
                        self.player2.update_episode_buffer(s, a, r, s1, d, v)

                    s = s1
                    if len(self.player1.episode_buffer) == 5 and not d:
                        v1 = sess.run(self.player1.local_A3C.value,
                                      feed_dict={
                                          self.player1.local_A3C.inputs: [s],
                                          self.player1.local_A3C.state_in[0]:
                                              rnn_state[0],
                                          self.player1.local_A3C.state_in[1]:
                                              rnn_state[1]})[0, 0]
                        self.player1.train(self.player1.episode_buffer,
                                           sess, gamma, v1)
                        self.player1.reset_episode_buffer()
                        sess.run(self.player1.update_local_ops)
                    if len(self.player2.episode_buffer) == 5 and not d:
                        v2 = sess.run(self.player2.local_A3C.value,
                                      feed_dict={
                                          self.player2.local_A3C.inputs: [s],
                                          self.player2.local_A3C.state_in[0]:
                                              rnn_state[0],
                                          self.player2.state_in[1]:
                                              rnn_state[1]})[0, 0]
                        self.player2.train(self.player2.episode_buffer,
                                           sess, gamma, v2)
                        self.player2.reset_episode_buffer()
                        sess.run(self.player2.update_local_ops)
                    if d:
                        break

                if len(self.player1.episode_buffer) > 0:
                    self.player1.train(self.player1.episode_buffer,
                                       sess, gamma, 0.0)
                if len(self.player2.episode_buffer) > 0:
                    self.player2.train(self.player2.episode_buffer,
                                       sess, gamma, 0.0)

                # TODO implement model saving
