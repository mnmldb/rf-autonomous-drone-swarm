from pydantic import BaseSettings

class EnvironmentSettings(BaseSettings):
    # path size
    path_size: int = 10
    # number of agents
    n_agents: int = 1

class QFuncSettings(BaseSettings):
    # hidden layer
    n_mid: int = 8
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
    policy_type: str = 'e_greedy'
    # epsilon
    eps_start: float = 1
    eps_end: float = 0
    r: float = 0.9999 # decay
    # stuck threshold
    max_stuck: int = 10000000 # pseudo large number to ignore stuck handling

class TrainingSettings(BaseSettings):
    # number of episodes
    train_episodes: int = 200000
    # max steps in each episode
    max_steps: int = 10 * 10
    # frequencty to copy Q function
    q_func_freq: int = 100
    # frequency to print progress
    print_progress_freq: int = 100