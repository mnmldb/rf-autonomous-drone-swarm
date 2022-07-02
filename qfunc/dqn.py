# referring to https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import Tuple

class Net(nn.Module):
    def __init__(self, n_obs: int, n_mid: int, n_action: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, n_mid) 
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer():
    def __init__(self, buffer_limit: int) -> None:
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque(maxlen=self.buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n: int):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class QValues:
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
    
class Policy:
    def __init__(self, policy_type: str, n_actions: int) -> None:
        self.policy_type = policy_type
        if self.policy_type not in ('e_greedy', 'softmax', 'greedy'):
            raise ValueError("Received invalid policy type={}".format(self.type))
        
        self.n_actions = n_actions
        self.action_idx = [i for i in range(self.n_actions)]

    def init_e_greedy(self, eps_start: float, eps_end: float, r: float) -> None:
        '''initialize eplison greedy parameters'''
        self.eps = eps_start
        self.eps_end = eps_end # lower bound of epsilon
        self.r = r # decrement rate of epsilon
    
    def update_eps(self) -> None:
        if self.eps > self.eps_end:
            self.eps *= self.r

    def _fnc_softmax(self, a: np.ndarray) -> np.ndarray:
        # deal with overflow
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y        

    def get_action(self, q_values: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.policy_type == 'e_greedy':
            try:
                if np.random.rand() < self.eps:
                    action = random.choice(self.action_idx)
                    greedy = False
                else:  # make decision by the highest q value
                    action = np.argmax(q_values)
                    greedy = True
            except Exception:
                raise NameError('Epsilon is not defined.')
        elif self.policy_type == 'softmax':
            p = self._fnc_softmax(q_values) # convert to probability
            action = np.random.choice(np.arange(self.n_actions), p=p)
            greedy = False
        elif self.policy_type == 'greedy':
            action = np.argmax(q_values)
            greedy = True
        
        return action, greedy



    