from typing import List, Dict

import torch
from torch.utils.data import Dataset

from dependency_parsing import sentence_to_matrix


class TranslationDetectionDataset(Dataset):
    def __init__(self, data: List[Dict], label_mapping: Dict[str, int],):
        self.data = data
        self.label_mapping = label_mapping
        self.target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples):
        text_tensor =  torch.stack([sentence_to_matrix(sentence['text']) for sentence in samples])
        label_tensor = torch.LongTensor([self.label_mapping[sentence['language']] for sentence in samples])
        return {'text': text_tensor.to(self.target_device),
                'label': label_tensor.to(self.target_device)}
