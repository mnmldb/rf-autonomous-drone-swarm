import logging
import random
import numpy as np
import copy
from typing import Tuple

from environment.grid_decentralized import Grid
from qfunc.idqn import QFunctions
from qfunc.utils_dqn.policy import Policy
from config.setting_grid_decentralized_dqn import EnvironmentSettings, QFuncSettings, PolicySettings, TrainingSettings

logging.basicConfig(
    filename='./logs/logfile.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)

env_settings = EnvironmentSettings()
qfunc_settings = QFuncSettings()
pol_settings = PolicySettings()
train_settings = TrainingSettings()

def _setup() -> Tuple[Grid, QFunctions, Policy]:
    ''' create environment and Q function '''
    env = Grid(grid_size=env_settings.grid_size, 
               n_agents=env_settings.n_agents, 
               agent_memory=env_settings.agent_memory)
    q = QFunctions(n_agents=env.n_agents,
                n_obs=env.n_observations,
                n_mid=qfunc_settings.n_mid, 
                n_action=env.n_actions, 
                is_gpu=qfunc_settings.is_gpu, 
                gamma=qfunc_settings.gamma, 
                lr=qfunc_settings.lr, 
                buffer_limit=qfunc_settings.buffer_limit)
    pol = Policy(policy_type=pol_settings.policy_type, 
                 n_actions=env.n_actions,
                 max_stuck=pol_settings.max_stuck)
    pol.init_e_greedy(eps_start=pol_settings.eps_start, 
                      eps_end=pol_settings.eps_end, 
                      r=pol_settings.r)
    return env, q, pol

def _train(env: Grid, q: QFunctions, pol: Policy) -> None:
    ''' records for each episode '''
    time_steps = [] # number of time steps in total
    epsilons = [] # epsilon at the end of each episode
    greedy = [] # the ratio of greedy choices
    coverage = [] # the ratio of visited cells at the end
    speed = [] # number of time steps to cover decent amount of cells
    sum_q_values = [] # sum of q-values
    results_mapping = [] # mapping status
    results_count = [] # count status
    total_reward = []
    q_class = [] 

    ''' execute training '''
    for episode in range(train_settings.train_episodes):
        state = env.reset()
        eps_tmp = pol.eps

        greedy_count = [0] * env.n_agents
        coverage_track = True
        epi_reward = [0] * env.n_agents

        for step in range(train_settings.max_steps):
            action_order = random.sample(env.idx_agents, env.n_agents) # return a random order of the drone indices
            for agent_idx in action_order:
                agent_obs = np.array(state[agent_idx])
                agent_q_values = q.get_q_values(agent_idx=agent_idx, obs=agent_obs) # get q values with the DQN network
                action, greedy_tf = pol.get_action(q_values=agent_q_values, stuck_count=env.stuck_counts[agent_idx])
                next_state, reward, done = env.step(action, agent_idx)

                done_mask = 0.0 if done else 1.0

                agent_next_obs = np.array(next_state[agent_idx])
                q.memory[agent_idx].put((agent_obs, action, reward, agent_next_obs, done_mask))
                epi_reward[agent_idx] += reward
                greedy_count[agent_idx] += greedy_tf * 1

                if done:
                    break
                
                # update the observation
                state = next_state

            # update epsilon
            pol.update_eps()

            # training
            for agent_idx in range(env.n_agents):
                if q.memory[agent_idx].size() > qfunc_settings.batch_size:
                    q.train(agent_idx=agent_idx, batch_size=qfunc_settings.batch_size)
        
            # update the target network
            if step % qfunc_settings.target_interval == 0:
                for agent_idx in range(env.n_agents):
                    q.update_target(agent_idx=agent_idx)

            # check if decent amoung of cells are visited
            current_coverage = env.get_coverage()
            if current_coverage >= env_settings.coverage_threshold and coverage_track:
                speed.append(step)
                coverage_track = False

            # check if the task is completed
            if done:
                time_steps.append(step)
                break
            elif step == train_settings.max_steps - 1:
                time_steps.append(step)
                if coverage_track:
                    speed.append(np.nan)

        # record
        epsilons.append(eps_tmp)
        coverage.append(env.get_coverage())
        greedy.append(list(map(lambda x: x / (step + 1), greedy_count)))
        sum_q_values.append([0])
        results_mapping.append(env.grid_status)
        results_count.append(env.grid_counts)
        total_reward.append(sum(epi_reward))

        if episode % train_settings.q_func_freq == 0:
            q_class.append(copy.deepcopy(q))

        logger.info('Episode {0} | Epsilon: {1:.3f} | Steps: {2} | Greedy Choices (%): {3:.3f} | Coverage (%): {4:.3f} | Steps to Visit {5}% Cells: {6} | Sum of Q-Values: {7:.1f} | Total Reward: {8}'
              .format(episode+1,
                      eps_tmp,  
                      step+1, 
                      np.mean(greedy[episode]), 
                      coverage[episode], 
                      env_settings.coverage_threshold * 100, 
                      speed[episode], 
                      sum_q_values[episode][0],
                      np.mean(total_reward[episode])))
        
        if episode % train_settings.print_progress_freq == 0:
            print('Episode {} finished.'.format(episode))

if __name__ == '__main__':
    logger.info('Simulation Start')
    env, q, pol = _setup()
    _train(env=env, q=q, pol=pol)
    print('Simulating')
    logger.info('Simulation End')

