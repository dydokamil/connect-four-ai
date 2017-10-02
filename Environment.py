import threading
import time

from Agent import Agent
from constants import EPS_START, EPS_STEPS, EPS_STOP, THREAD_DELAY


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, env, render=False, eps_start=EPS_START,
                 eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = env
        self.agent = Agent()

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render:
                self.env.render()

            a = self.agent.act(s)
            s_, r, done = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True
