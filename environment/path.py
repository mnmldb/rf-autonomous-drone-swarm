import gym
from gym import spaces
from space.action_space import MultiAgentActionSpace
from space.observation_space import MultiAgentObservationSpace

import numpy as np
import random
import copy
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Path(gym.Env):
    # action id
    XM = 0 # x minus
    XP = 1 # x plus
    
    def __init__(self, path_size: int, n_agents: int) -> None:
        super(Path, self).__init__()
        
        # size of 1D grid
        self.path_size = path_size

        # number of agents
        self.n_agents = n_agents
        self.idx_agents = list(range(n_agents)) # [0, 1, 2, ..., n_agents - 1]

        # initialize the position of the agent
        self.init_agent()
        
        # define action space
        self.n_actions = 2 # LEFT, RIGHT
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])
        logger.info('Action space is created.')
        
        # define observation space (fielf of view)
        self.obs_low = np.zeros(2) # (0, 0)
        self.obs_high = np.ones(2) * (path_size - 1) # (path_size - 1, path_size - 1)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(np.float32(self.obs_low), np.float32(self.obs_high)) for _ in range(self.n_agents)])
        self.n_observations = self.observation_space[0].shape[0] # https://github.com/openai/gym/blob/master/gym/spaces/box.py
        logger.info('Observation space is created.')

        logger.info('Environment successfully created.')
    
    def init_agent(self) -> None:
        ''' the agent starts from the origin'''
        self.agent_pos = [[0, 0] for i in range(self.n_agents)]
    
    def get_agent_obs(self) -> List[int]:
        return self.agent_pos

    def reset(self) -> List[int]:
        # initialize the position of the agent
        self.init_agent()
        return self.get_agent_obs()
        
    def step(self, agent_action: int, agent_idx: int) -> Tuple[List[np.ndarray], int, bool]: # i: index of the drone
        # original position
        org_x  = copy.deepcopy(self.agent_pos[agent_idx][0])

        # move the agent
        if agent_action == self.XM:
            self.agent_pos[agent_idx][0] -= 1
        elif agent_action == self.XP:
            self.agent_pos[agent_idx][0] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(agent_action))
        
        # account for the boundaries of the grid
        if self.agent_pos[agent_idx][0] > self.path_size - 1 or self.agent_pos[agent_idx][0] < 0:
            self.agent_pos[agent_idx][0] = org_x

        # reach the goal
        if self.agent_pos[agent_idx][0] == self.path_size - 1:
            reward = 10
            done = True
        else:
            reward = 0
            done = False
        
        return self.get_agent_obs(), reward, done
