import numpy as np
import torch
from torch.autograd import Variable

from ConnectFourEnvironment import ConnectFourEnvironment


class EnvironmentManager:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.global_episode_count = 0
        self.request_stop = False

        self.environment = ConnectFourEnvironment(play_with_rng=False)

    def work(self):
        print(f"Starting environment with agents {self.agent1.name} "
              f"and {self.agent2.name}")
        while not self.request_stop:
            s = self.environment.reset()
            d = False

            # self.agent1.reset()
            # self.agent2.reset()

            while not d:
                if self.environment.yellows_turn():
                    agent = self.agent1
                else:
                    agent = self.agent2

                s_variable = Variable(torch.from_numpy(s)).float().unsqueeze(0)
                policy, v, e = agent.choose_action(s_variable)
                # v = v.data.numpy().squeeze()
                policy = np.squeeze(policy.data.numpy())
                a = np.random.choice(policy, p=policy)
                a = np.argmax(policy == a)
                s1, r, d, info = self.environment.step(a)
                r /= 100.
                if d:
                    s1 = s

                agent.add_transition(s, a, r, v)
                s = s1

            self.global_episode_count += 1
            self.agent1.train(gamma=.99)
            self.agent2.train(gamma=.99)

            # if self.global_episode_count % 250 == 0:
            #     if self.agent1.name == 'agent0':
            #         self.agent1.save_model(self.global_episode_count)
            #     if self.agent2.name == 'agent1':
            #         self.agent2.save_model(self.global_episode_count)
