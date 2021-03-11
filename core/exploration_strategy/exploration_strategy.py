import abc


class ExplorationStrategy(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action_idx(self, possible_actions):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def stats(self):
        pass

    def end_epoch(self):
        pass
