import os
import threading
import time

import torch
import torch.optim as optim

from A3C_Network import Net
from Agent import Agent
from EnvironmentManager import EnvironmentManager
from common import s_size

if __name__ == '__main__':
    model_yellow = "yellow_model"
    model_red = "red_model"
    max_episode_length = s_size
    gamma = .99

    load_model = False
    if os.path.isfile(model_yellow) and os.path.isfile(model_red):
        load_model = True

    global_episodes = 0
    if load_model:
        print("Loading models from files...")
        red_network = torch.load(model_red)
        yellow_network = torch.load(model_yellow)
        print("Loaded.")
    else:
        red_network = Net()
        yellow_network = Net()

    # num_workers = multiprocessing.cpu_count()
    num_workers = 1

    managers = []

    yellow_optim = optim.Adam(yellow_network.parameters())
    red_optim = optim.Adam(red_network.parameters())

    num = 0
    for i in range(num_workers):
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
