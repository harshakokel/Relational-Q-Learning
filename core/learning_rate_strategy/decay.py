from core.learning_rate_strategy import LearningRateStrategy


class LinearDecay(LearningRateStrategy):

    def __init__(self, alpha=0.1, seed=None, min_alpha=0.01, num_epochs=100, decay_till=0.1):
        """Linear Decay

        Learning Rate begins from alpha and for (num_epochs * decay_till) iterations,
        alpha will be reduced by constant till it reaches min_alpha."""
        self.alpha = alpha
        self.init_alpha = alpha
        self.min_alpha = min_alpha
        self.num_epoch = num_epochs
        self.explore_ration = decay_till
        self.alpha_decay = (alpha - min_alpha) / (num_epochs * decay_till)

    def reset(self):
        self.alpha = self.init_alpha

    def end_epoch(self):
        if self.alpha > self.min_alpha:
            self.alpha -= self.alpha_decay
        else:
            self.alpha = self.min_alpha
        return self.alpha
