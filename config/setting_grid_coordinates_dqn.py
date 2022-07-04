from pydantic import BaseSettings

class EnvironmentSettings(BaseSettings):
    # grid size
    grid_size: int = 5
    # number of agents
    n_agents: int = 1
    # performance evaluation
    coverage_threshold: float = 0.9

class QFuncSettings(BaseSettings):
	# hidden layer
    n_mid: int = 16
    # parameters update
    gamma: float = 0.5 # discount factor
    lr: float = 0.001 # learning rate - https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
    # target update
    target_interval: int = 10
    # experiece replay
    buffer_limit: int = 100
    batch_size: int = 10
    # GPU
    is_gpu: bool = False

class PolicySettings(BaseSettings):
    # policy type
    policy_type: str = 'softmax'
    # epsilon
    eps_start: float = 1
    eps_end: float = 0
    r: float = 0.99999 # decay
    # stuck threshold
    max_stuck: int = 5

class TrainingSettings(BaseSettings):
    # number of episodes
    train_episodes: int = 100000
    # max steps in each episode
    max_steps: int = 5 * 5 * 2
    # frequencty to copy Q function
    q_func_freq: int = 100
    # frequency to print progress
    print_progress_freq: int = 100