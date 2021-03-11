from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=None):
        """Experience Replay Buffer"""
        self._buffer = deque(maxlen=max_size)

    def add_trajectory(self, traj):
        self._buffer.append(traj)

    def add_all_trajectories(self, trajectories):
        self._buffer += trajectories

    def get_trajectories(self, sample_size=None):
        if sample_size is None:
            return  self._buffer.copy()
        else:
            return [self._buffer[i] for i in (np.random.random(sample_size) * self.size).astype(int)]

    def get_diagnostics(self):
        return {'size':self.size}

    @property
    def size(self):
        return len(self._buffer)
    
class SuccessBuffer:
    def __init__(self, max_size=None):
        """Experience Replay Buffer"""
        self._buffer = deque(maxlen=max_size)

    def add_trajectory(self, traj):
        if traj[-1][1] == "SUCCESS":
            self._buffer.append(traj)

    def add_all_trajectories(self, trajectories):
        for traj in trajectories:
            self.add_trajectory(traj)


    def get_trajectories(self, sample_size=None):
        if sample_size is None:
            return  self._buffer.copy()
        else:
            return [self._buffer[i] for i in (np.random.random(sample_size) * self.size).astype(int)]

    def get_diagnostics(self):
        return {'size':self.size}

    @property
    def size(self):
        return len(self._buffer)