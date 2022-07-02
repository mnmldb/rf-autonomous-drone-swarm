import gym
from gym import spaces
from space.action_space import MultiAgentActionSpace
from space.observation_space import MultiAgentObservationSpace

import numpy as np
import random
import copy
import logging
from typing import List, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class Grid(gym.Env):
    ''' action id '''
    XM = 0 # x minus
    XP = 1 # x plus
    YM = 2 # y minus
    YP = 3 # y plus

    ''' cell status '''
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
        self.n_actions = 4
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
        self.n_observations = self.observation_space[0].shape[0] # https://github.com/openai/gym/blob/master/gym/spaces/box.py
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
        self.agent_action_history = [deque([no_action] * self.agent_memory) for i in range(self.n_agents)]

        # initialize the stuck count
        self.stuck_counts = [0] * self.n_agents

        logger.info('Agents are initialized.')

    def _compute_single_agent_obs(self, agent_idx: int) -> List[int]:
        # action histories
        self_action_history = list(self.agent_action_history[agent_idx]) # convert from queue to list
        others_action_history = [list(v) for k, v in enumerate(self.agent_action_history) if k != agent_idx]

        # relative positions
        origin_pos = self.agent_pos[agent_idx]
        absolute_pos = [v for k, v in enumerate(self.agent_pos) if k != agent_idx]
        relative_pos = (np.array(absolute_pos) - np.array(origin_pos)).tolist() # (1) convert to ndarray (2) compute relative pos by broadcasting (3) convert to lis
        
        # flatten and concatenate
        agent_observation = self_action_history + \
            [x for xs in others_action_history for x in xs] + \
            [x for xs in relative_pos for x in xs]
        
        return agent_observation

    def _get_agent_obs(self) -> List[List[int]]:
        ''' TEMPORARY: Need logic'''
        # self.agent_obs = [np.array([0] * len(self.obs_low))] * self.n_agents
        self.agent_obs = []
        for i in range(self.n_agents):
            self.agent_obs.append(self._compute_single_agent_obs(agent_idx=i))

        return self.agent_obs
    
    def _update_agent_action_history(self, agent_action: int, agent_idx: int) -> None:
        self.agent_action_history[agent_idx].appendleft(agent_action)
        self.agent_action_history[agent_idx].pop()

    def get_coverage(self) -> float:
        self.mapped_poi = (self.grid_status == 1).sum()
        return self.mapped_poi / self.n_poi

    def reset(self, initial_pos: List[List[int]] = None) -> List[np.ndarray]:
        # initialize the mapping status
        self._init_grid()
        # initialize the agent positions and action histories
        self._init_agent(initial_pos=initial_pos)

        logger.info('Grid and agents are reset.')

        # return the agent observations
        return self._get_agent_obs()

    def step(self, agent_action: int, agent_idx: int) -> Tuple[List[np.ndarray], int, bool]:
        # original position
        org_x  = copy.deepcopy(self.agent_pos[agent_idx][0])
        org_y  = copy.deepcopy(self.agent_pos[agent_idx][1])

        # move the agent temporarily
        if agent_action == self.XM:
            self.agent_pos[agent_idx][0] -= 1
        elif agent_action == self.XP:
            self.agent_pos[agent_idx][0] += 1
        elif agent_action == self.YM:
            self.agent_pos[agent_idx][1] -= 1
        elif agent_action == self.YP:
            self.agent_pos[agent_idx][1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(agent_action))
        logger.info('The agent moves to the new position temporarily.')
        
        # account for the boundaries of the grid
        if self.agent_pos[agent_idx][0] > self.grid_size - 1 or self.agent_pos[agent_idx][0] < 0 or self.agent_pos[agent_idx][1] > self.grid_size - 1 or self.agent_pos[agent_idx][1] < 0:
            self.agent_pos[agent_idx][0] = org_x
            self.agent_pos[agent_idx][1] = org_y 
            self.grid_counts[self.agent_pos[agent_idx][0], self.agent_pos[agent_idx][1]] += 1
            agent_reward = 0
            self._update_agent_action_history(agent_action=agent_action, agent_idx=agent_idx)
            logger.info('The agent goes back to the original position because the new position is out of the grid.')
        else:
            # previous status of the cell
            prev_status = self.grid_status[self.agent_pos[agent_idx][0], self.agent_pos[agent_idx][1]]
            if prev_status == self.NMAP:
                self.grid_counts[self.agent_pos[agent_idx][0], self.agent_pos[agent_idx][1]] += 1
                self.grid_status[self.agent_pos[agent_idx][0], self.agent_pos[agent_idx][1]] = 1
                agent_reward = 10
                self._update_agent_action_history(agent_action=agent_action, agent_idx=agent_idx)
                logger.info('The agent position is fixed. Reward: {}'.format(agent_reward))
            elif prev_status == self.MAP:
                self.grid_counts[self.agent_pos[agent_idx][0], self.agent_pos[agent_idx][1]] += 1
                agent_reward = 0
                self._update_agent_action_history(agent_action=agent_action, agent_idx=agent_idx)
                logger.info('The agent position is fixed. Reward {}'.format(agent_reward))

        # update the stuck count
        if org_x == self.agent_pos[agent_idx][0] and org_y == self.agent_pos[agent_idx][1]: # stuck
            self.stuck_counts[agent_idx] += 1
            logger.info('The agent stuck count is updated. Count: {}'.format(self.stuck_counts[agent_idx]))
        else:
            self.stuck_counts[agent_idx] = 0
            logger.info('The agent stuck count is reset.')

        # check if agents map all cells
        self.mapped_poi = (self.grid_status == 1).sum()
        done = bool(self.mapped_poi == self.n_poi)
        
        return self._get_agent_obs(), agent_reward, done