import numpy as np
import random

from model import Model
from experience_replay import ExperienceReplay

import torch
import torch.nn.functional as F
import torch.optim as optim

import config as c

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, s_size, a_size, seed):
        """

        Params
        ======
            s_size (int): dimension of each state
            a_size (int): dimension of each action
            seed (int): random seed
        """
        self.s_size = s_size
        self.a_size = a_size
        self.seed = random.seed(seed)

        # Initialize both the Q-networks
        self.local_dqn = Model(s_size, a_size, seed).to(device)
        self.target_dqn = Model(s_size, a_size, seed).to(device)
        self.optimizer = optim.Adam(self.local_dqn.parameters(), lr=c.LEARNING_RATE)

        # Initialize experience deque
        self.buffer = ExperienceReplay(a_size, c.REPLAY_BUFFER_SIZE, c.BATCH_SIZE, seed)

        # Time step counter used for updating as per UPDATE_FREQUENCY
        self.t_step = 0

    def step(self, s, a, r, s_next, done, transfer_method):
        # Add experience to dequeue
        self.buffer.add(s, a, r, s_next, done)

        # Learn if UPDATE_FREQUENCY matched.
        self.t_step = (self.t_step + 1) % c.UPDATE_FREQUENCY
        if self.t_step == 0:
            # Get random experiences to learn from.
            if len(self.buffer) > c.BATCH_SIZE:
                es = self.buffer.sample()
                self.learn(es, transfer_method, c.GAMMA)

    def act(self, state, transfer_method, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            isTransfer (int): 0 if pre-trained weights to be used, int otherwise
            eps (float): epsilon, for exploration
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_dqn.eval()
        with torch.no_grad():
            a_values = self.local_dqn(state, transfer_method)
        self.local_dqn.train()

        # Generate a random number. If larger than epsilon pick greedy, or random otherwise
        if random.random() > eps:
            return np.argmax(a_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.a_size))

    def learn(self, es, transfer_method, gamma):
        """Update parameters based on experiences.

        Params
        ======
            es (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        s_, a_, r_, s_next_, d_ = es

        # Max predicted Q-values
        target_Q_next = self.target_dqn(s_next_, transfer_method).detach().max(1)[0].unsqueeze(1)

        # Target Q-value
        target_Q = r_ + (gamma * target_Q_next * (1 - d_))

        # Expected Q-vales
        expected_Q = self.local_dqn(s_, transfer_method).gather(1, a_)

        loss = F.mse_loss(expected_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        update(self.local_dqn, self.target_dqn, c.TAU)


def update(local_dqn, target_dqn, tau):
    """Update weights.
    target_weights = tau*local_weights + (1 - tau)*target_weights

    Parameters:
        local_dqn (PyTorch model): weights will be copied from
        target_dqn (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_weights, local_weights in zip(target_dqn.parameters(), local_dqn.parameters()):
        target_weights.data.copy_(tau * local_weights.data + (1.0 - tau) * target_weights.data)
