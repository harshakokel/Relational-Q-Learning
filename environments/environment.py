from abc import abstractmethod
from copy import deepcopy
from itertools import chain, combinations

import numpy as np


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class Environment:

    def __init__(self, state):
        self._state = state

    @abstractmethod
    def apply_action(self, a):
        pass

    @abstractmethod
    def all_actions(self, state=None):
        """Provide list of all possible actions in state"""
        pass

    # @abstractmethod
    # def cost(self) -> float:
    #     pass

    @abstractmethod
    def observe(self, s=None, state_id=None):
        """Return observation for current state (converts state to facts)"""
        pass

    @abstractmethod
    def reset(self, state=None):
        pass

    @abstractmethod
    def modes(self):
        """Return mode for the domain"""
        pass

    def target(self):
        """Return target predicate for the domain"""
        return "move"

    @abstractmethod
    def is_goal_state(self, state=None):
        pass

    @abstractmethod
    def get_reward(self, state, action, next_state=None):
        pass

    @abstractmethod
    def env_name(self):
        pass

    @property
    def state(self):
        return deepcopy(self._state)

    @abstractmethod
    def get_solved_percent(self):
        pass

    @staticmethod
    def get_state_space():
        pass

    @staticmethod
    def get_diagnostics(paths, **kwargs):
        successes = [p['is_success'] for p in paths]
        rewards = [p['episode_reward'] for p in paths]
        percent_solved = [p['percent_solved'] for p in paths]
        average_reward = np.mean(rewards)
        reward_max = np.max(rewards)
        success_rate = sum(successes) / len(successes)
        lengths = [p['episode_length'] for p in paths]
        length_rate = sum(lengths) / len(lengths)
        return {'Success Rate': success_rate,
                'Episode length Mean': length_rate,
                'Episode length Min': min(lengths),
                'Episode counts': len(paths),
                'Percent Solved Mean': np.mean(percent_solved),
                'Percent Solved Max': np.max(percent_solved),
                'Total Reward Mean': average_reward,
                'Total Reward Max': reward_max}


class Task:

    @abstractmethod
    def is_goal_state(self, state):
        """Checks if current state is goal state for the task"""
        pass

    @abstractmethod
    def get_reward(self, state, action, next_state):
        """Returns reward for s, a, s'"""
        pass


class State:

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass
