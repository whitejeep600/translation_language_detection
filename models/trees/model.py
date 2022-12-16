import torch.nn as nn


class TranslationDetector(nn.Module):
    def __init__(self, representation_dim, sequence_length, num_classes):
        super(TranslationDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(sequence_length, 1),  # converting sentence matrix representation to vector representation
            nn.Linear(representation_dim, representation_dim),  # first layer, preserving dimension
            nn.Sigmoid(),
            nn.Linear(representation_dim, representation_dim),  # second layer, preserving dimension
            nn.Sigmoid(),
            nn.Linear(representation_dim, num_classes),  # conversion to logits
        )

    def forward(self, batch):
        return self.network(batch)
