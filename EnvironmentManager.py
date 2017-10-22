from ConnectFourEnvironment import ConnectFourEnvironment


class EnvironmentManager:
    def __init__(self, agent1, agent2, session_common, coord):
        self.agent1 = agent1
        self.agent2 = agent2
        self.session_common = session_common
        self.coord = coord
        self.rnn_state_yellow = None
        self.rnn_state_red = None
        self.global_episode_count = 0

        self.environment = ConnectFourEnvironment()

    def work(self):
        while not self.coord.should_stop():
            s = self.environment.reset()
            d = False

            self.agent1.reset()
            self.agent2.reset()

            while not d:
                if self.environment.yellows_turn():
                    agent = self.agent1
                else:
                    agent = self.agent2

                a, v = agent.choose_action(s)
                s1, r, d, info = self.environment.step(a)
                r /= 100.
                if d:
                    s1 = s

                agent.add_transition([s, a, r, v[0, 0]])
                s = s1

            self.global_episode_count += 1
            if self.global_episode_count % 250 == 0:
                if self.agent1.name == 'agent0':
                    self.agent1.save_model(self.global_episode_count)
                if self.agent2.name == 'agent1':
                    self.agent2.save_model(self.global_episode_count)


                    # if agent == self.agent1:
                    #     if info == 'prohibited':
                    #         self.agent2.add_reward(0)
                    #     else:
                    #         self.agent2.add_reward(-episode_reward)
                    # else:
                    #     if info == 'prohibited':
                    #         self.agent1.add_reward(0)
                    #     else:
                    #         self.agent1.add_reward(-episode_reward)
