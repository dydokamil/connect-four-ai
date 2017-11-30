import os

import torch
from torch.autograd import Variable

from ConnectFourEnvironment import ConnectFourEnvironment
from common import SAVE_DIR, NUM_PROCESSES, NUM_STACK

env = ConnectFourEnvironment()
model_path = os.path.join(SAVE_DIR, 'a2c', 'ConnectFourRed.pt')
actor_critic, _ = torch.load(model_path)

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * NUM_STACK, *obs_shape[1:])

current_obs = torch.zeros(1, *obs_shape)


def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if NUM_STACK > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


obs = env.reset()
update_current_obs(obs)

while True:
    s = env.reset()
    d = False
    while not d:
        if not env.yellows_turn():
            value, action, action_log_prob, states = actor_critic.act(
                Variable(current_obs, volatile=True),
                Variable(torch.zeros([1, 6, 7]), volatile=True),
                Variable(torch.ones([1, 6, 7]), volatile=True)
            )

            a = action.data.squeeze(1).cpu().numpy()
        else:
            a = input('Choose action (0-6): ')
            a = int(a)

        s, r, d, _ = env.step(a)
        env.render()
        update_current_obs(s)
    print("Resetting...")
