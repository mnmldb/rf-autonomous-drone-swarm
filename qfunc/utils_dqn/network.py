# referring to https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_obs: int, n_mid: int, n_action: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, n_mid) 
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc4 = nn.Linear(n_mid, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x