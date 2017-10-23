import os
import threading
import time

import torch.optim as optim

from A3C_Network import Net
from Agent import Agent
from EnvironmentManager import EnvironmentManager
from common import s_size

if __name__ == '__main__':
    model_yellow = "model_yellow"
    model_red = "model_red"
    max_episode_length = s_size
    gamma = .99

    load_model = False
    if not os.path.exists(model_yellow) and not os.path.exists(model_yellow):
        os.makedirs(model_yellow)
        os.makedirs(model_red)
    else:
        load_model = True

    global_episodes = 0
    red_network = Net()
    yellow_network = Net()

    # num_workers = multiprocessing.cpu_count()
    num_workers = 1

    managers = []

    yellow_optim = optim.Adam(yellow_network.parameters())
    red_optim = optim.Adam(red_network.parameters())

    for i in range(num_workers):
        num = 0
        agent1 = Agent(num, yellow_network, yellow_optim)
        agent2 = Agent(num + 1, red_network, red_optim)
        num += 2
        managers.append(EnvironmentManager(agent1, agent2))

    worker_threads = []

    for worker in managers:
        def worker_work(): worker.work()


        t = threading.Thread(target=worker_work)
        t.start()
        time.sleep(.5)
        worker_threads.append(t)

    for t in worker_threads:
        t.join()
