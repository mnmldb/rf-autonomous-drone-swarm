import numpy as np
import random
from typing import Tuple

class Policy:
    def __init__(self, policy_type: str, n_actions: int, max_stuck: int) -> None:
        self.policy_type = policy_type
        if self.policy_type not in ('e_greedy', 'softmax', 'greedy', 'random'):
            raise ValueError("Received invalid policy type={}".format(self.type))
        
        self.n_actions = n_actions
        self.action_idx = [i for i in range(self.n_actions)]

        self.max_stuck = max_stuck

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

    def get_action(self, q_values: np.ndarray, stuck_count: int) -> Tuple[np.ndarray, bool]:
        if stuck_count >= self.max_stuck:
            action = random.choice(self.action_idx)
            greedy = False
        else:
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
            elif self.policy_type == 'random':
                action = random.choice(self.action_idx)
                greedy = False
        
        return action, greedy