import abc


class LearningRateStrategy(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def end_epoch(self):
        pass
