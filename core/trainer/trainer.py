from abc import abstractmethod

class Trainer:

    @abstractmethod
    def train(self):
        """Train"""
        pass

    @abstractmethod
    def evaluate(self,  batch_size, q_function):
        pass

    @abstractmethod
    def get_training_batch(self,  batch_size, q_function):
        pass
