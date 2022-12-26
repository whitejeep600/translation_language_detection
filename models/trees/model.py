import torch
import torch.nn as nn

from utils import get_target_device


class InitialDimReduction(nn.Module):
    def __init__(self,representation_dim, sequence_length):
        super().__init__()
        self.vector = torch.nn.Parameter(torch.rand(sequence_length, device=get_target_device(), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.rand(representation_dim, device=get_target_device(), requires_grad=True))

    def forward(self, x):
        res = torch.matmul(x, self.vector) + self.bias
        return res


class TranslationDetector(nn.Module):
    def __init__(self, representation_dim, sequence_length, num_classes):
        super(TranslationDetector, self).__init__()
        self.network = nn.Sequential(
            InitialDimReduction(representation_dim, sequence_length),  # converting sentence matrix representation to vector representation
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim),  # second layer, preserving dimension
            nn.ReLU(),
            nn.Linear(representation_dim, num_classes),  # conversion to logits
        )

    def forward(self, batch):
        return self.network(batch)
