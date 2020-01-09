import numpy as np
import random

from collections import deque

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExperienceReplay:
    """Buffer class for storing experiences using deque"""

    def __init__(self, a_size, buffer_size, batch_size, seed):
        """
        Parameters:
            a_size (int): action space
            buffer_size (int): maximum buffer size
            batch_size (int): mini-batch size
            seed (int): random seed
        """
        self.a_size = a_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, s, a, r, s_next, done):
        """Add a new experience"""
        e = s, a, r, s_next, done
        self.buffer.append(e)

    def sample(self):
        """Sample random set of experiences"""

        es = random.sample(self.buffer, k=self.batch_size)
        s_, a_, r_, s_next_, d_ = [], [], [], [], []

        for e in es:
            if e:
                s_.append(e[0])
                a_.append(e[1])
                r_.append(e[2])
                s_next_.append(e[3])
                d_.append(e[4])

        s_ = torch.from_numpy(np.vstack(s_)).float().to(device)
        a_ = torch.from_numpy(np.vstack(a_)).long().to(device)
        r_ = torch.from_numpy(np.vstack(r_)).float().to(device)
        s_next_ = torch.from_numpy(np.vstack(s_next_)).float().to(device)
        d_ = torch.from_numpy(np.vstack(d_).astype(np.uint8)).float().to(device)

        return s_, a_, r_, s_next_, d_

    def __len__(self):
        """len of internal buffer."""
        return len(self.buffer)
