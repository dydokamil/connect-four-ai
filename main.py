import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.autograd import Variable

from ConnectFourEnvironment import ConnectFourEnvironment
from common import NUM_PROCESSES, NUM_STACK, CUDA, EPS, LR, ALPHA, \
    NUM_STEPS, NUM_FRAMES, VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM, \
    SAVE_DIR, SAVE_INTERVAL, USE_GAE, GAMMA, TAU
from model import MLPPolicy
from storage import RolloutStorage

NUM_UPDATES = NUM_FRAMES // NUM_STEPS // NUM_PROCESSES

if __name__ == '__main__':
    assert NUM_STEPS % 2 == 0
    action_shape = 1

    envs = [ConnectFourEnvironment for _ in range(NUM_PROCESSES)]
    envs = SubprocVecEnv(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * NUM_STACK, *obs_shape[1:])

    model_path = os.path.join(SAVE_DIR, 'a2c')
    if os.path.exists(model_path) and len(os.listdir(model_path)) == 2:
        print("Loading saved models...")
        actor_critic_yellow, _ = torch.load(
            os.path.join(model_path, 'ConnectFourYellow.pt'))
        actor_critic_red, _ = torch.load(
            os.path.join(model_path, 'ConnectFourRed.pt'))
    else:
        actor_critic_yellow = MLPPolicy(obs_shape[0], envs.action_space)
        actor_critic_red = MLPPolicy(obs_shape[0], envs.action_space)

    if CUDA:
        actor_critic_yellow.cuda()
        actor_critic_red.cuda()

    optim_yellow = optim.RMSprop(actor_critic_yellow.parameters(), lr=LR,
                                 eps=EPS, alpha=ALPHA)
    optim_red = optim.RMSprop(actor_critic_red.parameters(), lr=LR, eps=EPS,
                              alpha=ALPHA)

    rollouts_yellow = RolloutStorage(NUM_STEPS, NUM_PROCESSES, obs_shape,
                                     envs.action_space,
                                     actor_critic_yellow.state_size)
    rollouts_red = RolloutStorage(NUM_STEPS, NUM_PROCESSES, obs_shape,
                                  envs.action_space,
                                  actor_critic_red.state_size)

    current_obs = torch.zeros(NUM_PROCESSES, *obs_shape)


    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if NUM_STACK > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs


    obs = envs.reset()
    update_current_obs(obs)

    rollouts_yellow.observations[0].copy_(current_obs)
    rollouts_red.observations[0].copy_(current_obs)

    if CUDA:
        current_obs = current_obs.cuda()
        rollouts_yellow.cuda()
        rollouts_red.cuda()

    losses_yellow = []
    losses_red = []

    start = time.time()
    for j in range(int(NUM_UPDATES)):
        for step in range(NUM_STEPS):
            actor_critic = actor_critic_yellow if step % 2 == 0 \
                else actor_critic_red

            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts_red.observations[step],
                         volatile=True),
                Variable(rollouts_red.states[step], volatile=True),
                Variable(rollouts_red.masks[step], volatile=True)
            )

            cpu_actions = action.data.squeeze(1).cpu().numpy()

            obs, reward, done, info = envs.step(cpu_actions)
            prohibited_mask = np.where(reward == -6)
            reward_yellow = reward.copy()
            reward_red = reward.copy()

            if step % 2 == 0:
                reward_red[prohibited_mask] = 0
                reward_red *= -1
            else:
                reward_yellow[prohibited_mask] = 0
                reward_yellow *= -1

            reward_yellow = torch.from_numpy(
                np.expand_dims(np.stack(reward_yellow), 1)
            ).float()

            reward_red = torch.from_numpy(
                np.expand_dims(np.stack(reward_red), 1)
            ).float()

            masks = torch.FloatTensor(
                [[0.] if done_ else [1.] for done_ in done]
            )

            if CUDA:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts_yellow.insert(step, current_obs, states.data,
                                   action.data, action_log_prob.data,
                                   value.data, reward_yellow, masks)
            rollouts_red.insert(step, current_obs, states.data,
                                action.data, action_log_prob.data,
                                value.data, reward_red, masks)

        for actor_critic, rollouts, optimizer in [(actor_critic_yellow,
                                                   rollouts_yellow,
                                                   optim_yellow),
                                                  (actor_critic_red,
                                                   rollouts_red,
                                                   optim_red)]:
            next_value = actor_critic(
                Variable(rollouts.observations[-1], volatile=True),
                Variable(rollouts.states[-1], volatile=True),
                Variable(rollouts.masks[-1], volatile=True)
            )[0].data

            rollouts.compute_returns(next_value, USE_GAE, GAMMA, TAU)

            (values, action_log_probs, dist_entropy, states) = \
                actor_critic.evaluate_actions(
                    Variable(
                        rollouts.observations[:-1].view(-1, *obs_shape)
                    ),
                    Variable(
                        rollouts.states[0].view(-1, actor_critic.state_size)
                    ),
                    Variable(
                        rollouts.masks[:-1].view(-1, 1)
                    ),
                    Variable(
                        rollouts.actions.view(-1, action_shape)
                    )
                )

            values = values.view(NUM_STEPS, NUM_PROCESSES, 1)
            action_log_probs = action_log_probs.view(NUM_STEPS, NUM_PROCESSES,
                                                     1)
            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()
            if actor_critic == actor_critic_yellow:
                losses_yellow.append(value_loss.data[0])
            else:
                losses_red.append(value_loss.data[0])

            action_loss = -(
                    Variable(advantages.data) * action_log_probs
            ).mean()

            optimizer.zero_grad()
            (value_loss
             * VALUE_LOSS_COEF
             + action_loss
             - dist_entropy
             * ENTROPY_COEF).backward()

            nn.utils.clip_grad_norm(actor_critic.parameters(),
                                    MAX_GRAD_NORM)
            optimizer.step()

            rollouts.after_update()

            if j % SAVE_INTERVAL == 0 and SAVE_DIR != "":
                save_path = os.path.join(SAVE_DIR, 'a2c')
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if CUDA:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model,
                              hasattr(envs, 'ob_rms') and envs.ob_rms or None]

                torch.save(save_model,
                           os.path.join(save_path,
                                        f"ConnectFour{'Yellow' if actor_critic == actor_critic_yellow else 'Red'}.pt")
                           )

        if j % 100 == 1:
            print(f'Value loss yellow: {np.mean(losses_yellow)}')
            print(f'Value loss red: {np.mean(losses_red)}')
            losses_yellow.clear()
            losses_red.clear()
