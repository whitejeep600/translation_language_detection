import torch
import torch.nn as nn

from utils import get_target_device


class Convolution(nn.Module):
    def __init__(self, num_filters, representation_dim):
        super().__init__()
        self.filters = torch.nn.Parameter(torch.rand(num_filters, representation_dim)).to(get_target_device())
        self.biases = torch.nn.Parameter(torch.rand(num_filters)).to(get_target_device())
        # each filter corresponds to a row in the matrix, whose dot product with each word representation
        # is calculated. Afterwards, the maximum value for each filter is selected.

    def forward(self, x):
        res = torch.matmul(self.filters, x)
        maximum = torch.max(res, dim=2).values
        # row-wise max (but dim=2 because the first dimension is position in batch)
        with_bias = maximum + self.biases
        return with_bias


class TranslationDetector(nn.Module):
    def __init__(self, num_filters, representation_dim, num_classes):
        super(TranslationDetector, self).__init__()
        self.network = nn.Sequential(
            Convolution(num_filters, representation_dim),
            nn.Sigmoid(),
            nn.Linear(num_filters, num_filters),  # first layer, preserving dimension
            nn.Sigmoid(),
            nn.Linear(num_filters, num_filters),  # second layer, preserving dimension
            nn.Sigmoid(),
            nn.Linear(num_filters, num_classes),  # conversion to logits
        )

    def forward(self, batch):
        return self.network(batch)
