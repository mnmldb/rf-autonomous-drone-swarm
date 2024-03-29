# referring to https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .utils_dqn.network import Net
from .utils_dqn.replay_buffer import ReplayBuffer

class QFunction:
    def __init__(self, n_obs: int, n_mid: int, n_action: int, is_gpu: bool, gamma: float, lr: float, buffer_limit: int) -> None:
        ''' agents share the single DQN while training '''
		# network setup
        self.n_obs = n_obs  # observation space
        self.n_mid = n_mid
        self.n_action = n_action  # action space

        self.net_action = Net(n_obs=n_obs, n_mid=n_mid, n_action=n_action) # network to decide actions
        self.net_target = Net(n_obs=n_obs, n_mid=n_mid, n_action=n_action) # target network
        self.net_target.load_state_dict(self.net_action.state_dict()) # sync parameters
        self.is_gpu = is_gpu 
        if self.is_gpu:
            self.net_action.cuda()
            self.net_target.cuda()

        # training setup
        self.gamma = gamma  # discount rate
        self.lr = lr # learning rate
        self.loss_fnc = nn.MSELoss()
        self.optimizer = optim.Adam(self.net_action.parameters(), lr=self.lr)

        # experience replay setup
        self.buffer_limit = buffer_limit
        self.memory = ReplayBuffer(buffer_limit=self.buffer_limit)

    def train(self, batch_size: int) -> None:
        obs, action, reward, next_obs, done_mask = self.memory.sample(batch_size)

        if self.is_gpu:
            obs, action, reward, next_obs, done_mask = obs.cuda(), action.cuda(), reward.cuda(), next_obs.cuda(), done_mask.cuda()

        # current q-value in the online network
        q_out = self.net_action(obs) # batch_size x number of actions
        q_a = q_out.gather(1, action) # batch_size x 1 (select the corresponding q-value of each experience)

        # maximum q-value in the target network
        next_q_max = self.net_target(next_obs).max(1)[0].unsqueeze(1) # batch_size x 1

        # target
        target = reward + self.gamma * next_q_max * done_mask # batch_size x 1
            
        # difference between the current q-value and the target
        loss = self.loss_fnc(q_a, target) # 1 x 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self) -> None:
        self.net_target.load_state_dict(self.net_action.state_dict())

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs).float()
        if self.is_gpu:
            obs = obs.cuda()

        q_values = self.net_action.forward(obs) # make decision with the q network
        q_values = q_values.detach().cpu().numpy()
        return q_values