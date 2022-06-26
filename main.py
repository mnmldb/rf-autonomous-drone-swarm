import logging
import random
import numpy as np
import copy
from environment.grid import Grid
from qfunc.qtables import QTables
from config.setting import EnvironmentSettings, QTableSettings, TrainingSettings

logging.basicConfig(
    filename='./logs/logfile.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)

env_settings = EnvironmentSettings()
qfunc_settings = QTableSettings()
train_settings = TrainingSettings()

# records for each episode
time_steps = [] # number of time steps in total
epsilons = [] # epsilon at the end of each episode
greedy = [] # the ratio of greedy choices
coverage = [] # the ratio of visited cells at the end
speed = [] # number of time steps to cover decent amount of cells
sum_q_values = [] # sum of q-values
results_mapping = [] # mapping status
results_count = [] # count status
total_reward = []
total_action_values = []
total_greedy_action_values = []

q_class = []

def _train():
    for episode in range(train_settings.train_episodes):
        state = env.reset()
        state = [arr.astype('int') for arr in state] # convert from float to integer
        eps_tmp = q.eps

        greedy_count = [0] * env.n_agents
        coverage_track = True
        epi_reward = [0] * env.n_agents
        epi_action_value = [0] * env.n_agents
        epi_greedy_action_value = [0] * env.n_agents

        for step in range(train_settings.max_steps):
            action_order = random.sample(env.idx_agents, env.n_agents) # return a random order of the drone indices
            for agent_i in action_order:
                action, greedy_tf, action_value = q.get_action(observations=state, 
                                                               agent_i=agent_i, 
                                                               stuck_counts=env.stuck_counts, 
                                                               max_stuck=qfunc_settings.max_stuck, 
                                                               e_greedy=qfunc_settings.e_greedy, 
                                                               softmax=qfunc_settings.softmax)
                next_state, reward, done = env.step(action, agent_i)
                next_state = [arr.astype('int') for arr in next_state] # convert from float to integer
                q.train(state, next_state, action, reward, done, agent_i)

                epi_reward[agent_i] += reward
                greedy_count[agent_i] += greedy_tf * 1
                epi_action_value[agent_i] += action_value
                epi_greedy_action_value[agent_i] += action_value * greedy_tf

                if done:
                    break
            
                # update the observation
                state = next_state

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
        time_steps.append(step + 1)
        epsilons.append(eps_tmp)
        coverage.append(env.get_coverage())
        greedy.append(list(map(lambda x: x / (step + 1), greedy_count)))
        sum_q_values.append([q.q_tables[0].sum()])
        results_mapping.append(env.grid_status)
        results_count.append(env.grid_counts)
        total_reward.append(epi_reward)
        total_action_values.append(epi_action_value)
        total_greedy_action_values.append(epi_greedy_action_value)

        if episode % train_settings.q_func_freq == 0:
            q_class.append(copy.deepcopy(q))

        # update epsilon
        q.update_eps()

        logger.info('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices (%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Sum of Q-Values: {7:.1f},    Total Reward: {8}'\
            .format(episode+1, eps_tmp, step+1, np.mean(greedy[episode]), coverage[episode], env_settings.coverage_threshold * 100, speed[episode], sum_q_values[episode][0], np.mean(total_reward[episode])))
        
        if episode % train_settings.print_progress_freq == 0:
            print('Episode {} finished.'.format(episode))

if __name__ == '__main__':
    logger.info('Simulation Start')
    env = Grid(x_size=env_settings.x_size, 
               y_size=env_settings.y_size, 
               n_agents=env_settings.n_agents, 
               fov_x=env_settings.fov_x, 
               fov_y=env_settings.fov_y)
    q = QTables(observation_space=env.observation_space, 
                action_space=env.action_space, 
                eps_start=qfunc_settings.eps_start, 
                eps_end=qfunc_settings.eps_end, 
                gamma=qfunc_settings.gamma, 
                r=qfunc_settings.r, 
                lr=qfunc_settings.lr)
    _train()
    print('Simulating')
    logger.info('Simulation End')