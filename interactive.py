import os

import numpy as np
import torch
from torch.autograd import Variable

from Agent import Agent
from ConnectFourEnvironment import ConnectFourEnvironment

env = ConnectFourEnvironment(play_with_rng=False)

model_dir = './model_yellow'
model_path = os.path.join(model_dir, os.listdir(model_dir)[0])

model = torch.load(model_path)

agent = Agent(0, model, None)

while True:
    env.reset()
    d = False
    while not d:
        if env.yellows_turn():
            s = env.get_state().flatten()
            s_var = Variable(torch.from_numpy(s)).float().unsqueeze(0)
            a_dist, _, _ = agent.choose_action(s_var)
            a_dist = np.squeeze(a_dist.data.numpy())
            a = np.random.choice(a_dist, p=a_dist)
            a = np.argmax(a == a_dist)
        else:
            a = input('Choose action (0-6): ')
            a = int(a)

        _, r, d, _ = env.step(a)
        env.render()
