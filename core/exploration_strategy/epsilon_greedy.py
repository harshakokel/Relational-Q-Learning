from core.exploration_strategy import ExplorationStrategy
import random


class EpsilonGreedy(ExplorationStrategy):

    def __init__(self, epsilon=0.1, seed=None):
        self.epsilon = epsilon
        self.seed = seed
        self.rng = random.Random(seed)

    def get_action_idx(self, best_idx, action_space):
        if self.rng.random() <= self.epsilon:
            return self.rng.randrange(0, action_space)
        return best_idx

    def reset(self):
        self.rng = random.Random(self.seed)

    def stats(self):
        return {'epsilon': self.epsilon}


class EpsilonGreedyWithLinearDecay(EpsilonGreedy):

    def __init__(self, epsilon=1.0, seed=None, min_epsilon=0.1, num_epochs=100, explore_ratio=0.1):
        """Linear Decay

        Exploration begins from epsilon and for (num_epochs * explore_ratio) iterations,
        exploration will be reduced by constant till it reaches min_epsilon."""
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.num_epoch = num_epochs
        self.explore_ration = explore_ratio
        self.seed = seed
        self.rng = random.Random(seed)
        self.epsilon_decay = (epsilon - min_epsilon) / (num_epochs * explore_ratio)

    def reset(self):
        self.epsilon = self.init_epsilon
        self.rng = random.Random(self.seed)

    def end_epoch(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon


class EpsilonGreedyWithExponentialDecay(EpsilonGreedy):

    def __init__(self, epsilon=0.99, seed=None, min_epsilon=0.7, decay_rate=0.975):
        """Exponential Decay

        Exploration begins from epsilon and is reduced by factor of decay_rate
        till it reaches min_epsilon."""
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.rng = random.Random(seed)
        self.decay_rate = decay_rate

    def reset(self):
        self.epsilon = self.init_epsilon
        self.rng = random.Random(self.seed)

    def end_epoch(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay_rate
        else:
            self.epsilon = self.min_epsilon


class EpsilonGreedyWithHeuristicDecay(EpsilonGreedy):

    def __init__(self, epsilon=1.0, seed=None, min_epsilon=0.1):
        """Heuristic Decay

        Exploration begins from epsilon and is reduced by factor of 1+epsilon
        till it reaches min_epsilon."""
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self):
        self.epsilon = self.init_epsilon
        self.rng = random.Random(self.seed)

    def end_epoch(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon / (self.epsilon + 1)
        else:
            self.epsilon = self.min_epsilon
