# referring to https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .utils_dqn.network import Net
from .utils_dqn.replay_buffer import ReplayBuffer

class QFunctions:
    def __init__(self, n_agents: int, n_obs: int, n_mid: int, n_action: int, is_gpu: bool, gamma: float, lr: float, buffer_limit: int) -> None:
        ''' agents have their own DQN networks '''

        #----- Common Setup -----#
        # number of agents
        self.n_agents = n_agents

		# network
        self.n_obs = n_obs  # observation space
        self.n_mid = n_mid
        self.n_action = n_action  # action space
        self.is_gpu = is_gpu 

        # training
        self.gamma = gamma  # discount rate
        self.lr = lr # learning rate
        self.loss_fnc = nn.MSELoss()

        # experience replay
        self.buffer_limit = buffer_limit

        #----- Individual Setup -----#
        # network
        self.net_action = [] # network to decide actions
        self.net_target = [] # target network

        # training
        self.optimizer = []

        # experience replay
        self.memory = []

        # loop for all agents
        for agent_idx in range(n_agents):
            # network
            self.net_action.append(Net(n_obs=n_obs, n_mid=n_mid, n_action=n_action))
            self.net_target.append(Net(n_obs=n_obs, n_mid=n_mid, n_action=n_action))
            self.net_target[agent_idx].load_state_dict(self.net_action[agent_idx].state_dict()) # sync parameters
            
            if self.is_gpu:
                self.net_action[agent_idx].cuda()
                self.net_target[agent_idx].cuda()

            # training
            self.optimizer.append(optim.Adam(self.net_action[agent_idx].parameters(), lr=self.lr))

            # experience replay
            self.memory.append(ReplayBuffer(buffer_limit=self.buffer_limit))

    def train(self, agent_idx: int, batch_size: int) -> None:
        obs, action, reward, next_obs, done_mask = self.memory[agent_idx].sample(batch_size)

        if self.is_gpu:
            obs, action, reward, next_obs, done_mask = obs.cuda(), action.cuda(), reward.cuda(), next_obs.cuda(), done_mask.cuda()

        # current q-value in the online network
        q_out = self.net_action[agent_idx](obs) # batch_size x number of actions
        q_a = q_out.gather(1, action) # batch_size x 1 (select the corresponding q-value of each experience)

        # maximum q-value in the target network
        next_q_max = self.net_target[agent_idx](next_obs).max(1)[0].unsqueeze(1) # batch_size x 1

        # target
        target = reward + self.gamma * next_q_max * done_mask # batch_size x 1
            
        # difference between the current q-value and the target
        loss = self.loss_fnc(q_a, target) # 1 x 1
        self.optimizer[agent_idx].zero_grad()
        loss.backward()
        self.optimizer[agent_idx].step()

    def update_target(self, agent_idx: int) -> None:
        self.net_target[agent_idx].load_state_dict(self.net_action[agent_idx].state_dict())

    def get_q_values(self, agent_idx: int, obs: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs).float()
        if self.is_gpu:
            obs = obs.cuda()

        q_values = self.net_action[agent_idx].forward(obs) # make decision with the q network
        q_values = q_values.detach().cpu().numpy()
        return q_values