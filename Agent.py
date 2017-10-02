import numpy as np

from constants import NUM_ACTIONS, GAMMA_N, GAMMA, N_STEP_RETURN


class Agent:
    def __init__(self, brain):
        self.brain = brain

        self.memory = []  # used for n_step return
        self.R = 0.

    def act(self, s):
        s = np.array([s])
        p = self.brain.predict_p(s)[0]

        # a = np.argmax(p)
        a = np.random.choice(NUM_ACTIONS, p=p)

        return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        # turn action into one-hot representation
        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps,
            # the computation is incorrect
