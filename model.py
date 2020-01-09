import torch
import torch.nn as nn

import config as c

from initialize_weights import Init_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, s_size, a_size, seed, fc1_units=c.FC1_UNITS, fc2_units=c.FC2_UNITS):
        """3 Fully Connected Layers

        Parameters:
            s_size (int): Dimension of each state
            a_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Model, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(s_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, a_size)
        self.isTransfer = True  # Controls weight transfers over episodes

    def forward(self, state, isTransfer=0):
        """Forward pass for the state-action classification problem"""

        if self.isTransfer and isTransfer:
            fc1_shape = self.fc1.weight.data.shape[1]
            weights = Init_weights().weight_transfer(isTransfer, fc1_shape)
            if isTransfer == 2:
                self.fc2.weight.data = torch.nn.Parameter(weights, requires_grad=True)
            else:
                self.fc1.weight.data = torch.nn.Parameter(weights, requires_grad=True)

            self.isTransfer = False
        fwd = torch.tanh(self.fc1(state))
        fwd = torch.tanh(self.fc2(fwd))
        fwd = self.fc3(fwd)

        return fwd
