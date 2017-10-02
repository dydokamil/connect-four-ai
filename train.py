# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import time

# -- constants
from Brain import Brain
from ConnectFourEnvironment import ConnectFourEnvironment
from Environment import Environment
from Optimizer import Optimizer
# -- main
from constants import THREADS, OPTIMIZERS, RUN_TIME

env_test = Environment(ConnectFourEnvironment(),
                       render=True, eps_start=0., eps_end=0.)

brain = Brain()  # brain is global in A3C

envs = [Environment(ConnectFourEnvironment()) for _ in range(THREADS)]
opts = [Optimizer(brain=brain) for _ in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Saving the model...")
brain.save_model()

print("Training finished")
# env_test.run()
