import torch
import torch.nn as nn


class InitialDimReduction(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.vector = torch.nn.Parameter(torch.rand(sequence_length)).to("cuda:0")
        assert self.vector.requires_grad

    def forward(self, x):
        res = torch.matmul(x, self.vector)
        return res


class TranslationDetector(nn.Module):
    def __init__(self, representation_dim, sequence_length, num_classes):
        super(TranslationDetector, self).__init__()
        self.network = nn.Sequential(
            InitialDimReduction(sequence_length),  # converting sentence matrix representation to vector representation
            nn.Linear(representation_dim, representation_dim),  # first layer, preserving dimension
            nn.Sigmoid(),
            nn.Linear(representation_dim, representation_dim),  # second layer, preserving dimension
            nn.Sigmoid(),
            nn.Linear(representation_dim, num_classes),  # conversion to logits
        )

    def forward(self, batch):
        return self.network(batch)
