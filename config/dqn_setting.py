from pydantic import BaseSettings

class EnvironmentSettings(BaseSettings):
    # grid size
    grid_size: int = 5
    # number of agents
    n_agents: int = 2
    # agents capability
    agent_memory: int = 5
    # performance evaluation
    coverage_threshold: float = 0.9

class QFuncSettings(BaseSettings):
	# hidden layer
    n_mid: int = 8
    # parameters update
    gamma: float = 0.9 # discount factor
    lr: float = 0.1 # learning rate
    # target update
    target_interval: int = 10
    # experiece replay
    buffer_limit: int = 10
    batch_size: int = 10
    # GPU
    is_gpu: bool = False
    # stuck threshold
    max_stuck: int = 100000

class PolicySettings(BaseSettings):
    # policy type
    policy_type: str = 'e_greedy'
    # epsilon
    eps_start: float = 1
    eps_end: float = 0
    r: float = 0.99 # decay

class TrainingSettings(BaseSettings):
    # number of episodes
    train_episodes: int = 200000
    # max steps in each episode
    max_steps: int = 10 * 10 * 2
    # frequencty to copy Q function
    q_func_freq: int = 1000
    # frequency to print progress
    print_progress_freq: int = 100


