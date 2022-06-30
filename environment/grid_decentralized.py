import gym
from gym import spaces
from space.action_space import MultiAgentActionSpace
from space.observation_space import MultiAgentObservationSpace

import numpy as np
import random
import copy
import logging
from typing import List

logger = logging.getLogger(__name__)

class Grid(gym.Env):
    ''' action id '''
    XM = 0 # x minus
    XP = 1 # x plus
    YM = 2 # y minus
    YP = 3 # y plus

    ''' cell status '''
    OOG = -1 # out of the grid
    NMAP = 0 # cells not mapped
    MAP = 1 # cells mapped
    
    def __init__(self, grid_size: int, n_agents: int, agent_memory: int) -> None:
        super(Grid, self).__init__()
        
        # size of 2D grid
        self.grid_size = grid_size

        # number of agents
        self.n_agents = n_agents
        self.idx_agents = list(range(n_agents)) # [0, 1, 2, ..., n_agents - 1]
        
        # define action space
        self.n_actions = 4 # LEFT, RIGHT, TOP, BOTTOM
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])
        logger.info('Action space is defined.')

        # define observation space consisting of:
        ## 1. past actions of the agent and 2. past actions of other agents
        self.agent_memory = agent_memory
        past_actions_low = np.zeros(self.agent_memory * self.n_agents)
        past_actions_high = np.ones(self.agent_memory * self.n_agents) * (self.n_actions - 1)
        ## 3. relative positions of other agents (x y coordinate)
        relative_pos_low = np.ones(2 * (self.n_agents - 1)) * (self.grid_size - 1) * (-1)
        relative_pos_high = np.ones(2 * (self.n_agents - 1)) * (self.grid_size - 1)
        ## concatenate
        self.obs_low = np.concatenate([past_actions_low, relative_pos_low])
        self.obs_high = np.concatenate([past_actions_high, relative_pos_high])
        self.observation_space = MultiAgentObservationSpace([spaces.Box(np.float32(self.obs_low), np.float32(self.obs_high)) for _ in range(self.n_agents)])
        logger.info('Observation space is created.')

        # initialize the mapping status
        self._init_grid()

        # initialize the agents' positions
        self._init_agent()

        logger.info('Environment is created.')

    def _init_grid(self) -> None:
        ''' initialize the mapping status '''
        ''' a cell can be occupied by several drones'''
        # -1: out of the grid
        # 0: cells not mapped
        # 1: cells mapped

        self.grid_status = np.zeros([self.grid_size, self.grid_size])
        self.grid_counts = np.zeros([self.grid_size, self.grid_size])
        self.n_poi = self.grid_size * self.grid_size

        logger.info('Grid is initialized.')
    
    def _init_agent(self, initial_pos: List[List[int]] = None) -> None:
        ''' initialize agents' positions and action histories '''
        # initialize positions
        self.agent_pos = []
        if initial_pos is not None:
            self.agent_pos = initial_pos
            for i in range(self.n_agents):
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = self.MAP
        else:
            # if initial positions are not specified, they are randomly determined
            for i in range(self.n_agents):
                agent_pos_x = random.randrange(0, self.grid_size)
                agent_pos_y = random.randrange(0, self.grid_size)
                self.agent_pos.append([agent_pos_x, agent_pos_y])
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = self.MAP
        
        # initialize action histories
        no_action = - 1 # place holder
        self.agent_action_history = [[no_action] * self.agent_memory] * self.n_agents

        # initialize the stuck count
        self.stuck_counts = [0] * self.n_agents

        logger.info('Agents are initialized.')
    
    def _get_agent_obs(self) -> List[np.ndarray]:
        ''' TEMPORARY: Need logic'''
        self.agent_obs = [np.array([0] * len(self.obs_low))] * self.n_agents

        return self.agent_obs

    def get_coverage(self) -> float:
        self.mapped_poi = (self.grid_status == 1).sum()
        return self.mapped_poi / self.n_poi
