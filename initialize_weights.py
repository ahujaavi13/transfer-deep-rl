import torch

import config as c

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IN_WEIGHTS_FILE = c.BASE_DIRECTORY + "/" + c.FILENAME  # Name of the file having weights to be transferred.


class Init_weights:
    """Transfer weights based on specific method."""

    def __init__(self):
        self.weights = torch.load(IN_WEIGHTS_FILE, map_location=torch.device('cpu'))

    def weight_transfer(self, transfer_method, fc1_shape):

        """
        Parameters:
            transfer_method (int): Value 0 to 6 for different methods.
            fc1_shape (): Shape of first hidden layer
        """

        # Architecture changes

        w = self.weights
        if transfer_method == 1:
            print("Weights Loaded for ", transfer_method)
            w = self.weights['fc2_.weight'].T
            ind = int(w.shape[0] / fc1_shape)
            x = torch.mean(w[:ind], dim=0).unsqueeze(1)
            for i in range(2, fc1_shape + 1):
                x = torch.cat((x, torch.mean(w[:ind], dim=0).unsqueeze(1)), dim=1)
            w = x

        elif transfer_method == 2:
            print("Weights Loaded for ", transfer_method)
            w = self.weights['fc2.weight']
            w = torch.cat((w, torch.zeros(144, 120).to(device)), dim=1)
            w = torch.cat((w, torch.zeros(48, 192).to(device)), dim=0)

        elif transfer_method == 3:
            print("Weights Loaded for ", transfer_method)
            w = self.weights['fc2.weight']
            w = torch.cat((w, torch.zeros(144, 120).to(device)), dim=1)
            ind = int(w.shape[0] / fc1_shape)
            x = torch.mean(w[:ind], dim=0).unsqueeze(1)
            for i in range(2, fc1_shape + 1):
                x = torch.cat((x, torch.mean(w[:ind], dim=0).unsqueeze(1)), dim=1)
            w = x

        # Information extraction

        elif transfer_method == 4:
            print("Weights Loaded for ", transfer_method)
            w = self.weights['fc3.weight'].T
            w = torch.cat((w, torch.zeros(w.shape).to(device)), dim=1)

        elif transfer_method == 5:
            print("Weights Loaded for ", transfer_method)
            w = self.weights['fc3.weight'].T
            w = torch.cat((w, w), dim=1)

        elif transfer_method == 6:
            print("Weights Loaded for ", transfer_method)
            w = self.weights['fc3.weight']
            w = repeat(w, [2, 1]).T

        return w


def repeat(t, dims):
    """

    parameters:
    t (tensor): Original tensor
    dims: dimensions
    """

    if len(dims) != len(t.shape):
        raise ValueError("Number of dimensions of tensor should be equal to length of dims")
    for index, dim in enumerate(dims):
        repeat_vector = [1] * (len(dims) + 1)
        repeat_vector[index + 1] = dim
        new_shape = list(t.shape)
        new_shape[index] *= dim
        t = t.unsqueeze(index + 1).repeat(repeat_vector).reshape(new_shape)
    return t
