from typing import List, Dict

import torch
from torch.utils.data import Dataset

from translation_language_detection.models.trees.dependency_parsing import sentence_to_matrix


class TranslationDetectionDataset(Dataset):
    def __init__(self, data: List[Dict], label_mapping: Dict[str, int],):
        self.data = data
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        return {'text': torch.FloatTensor([sentence_to_matrix(sentence['text']) for sentence in samples]),
                'label': torch.FloatTensor([self.label_mapping[sentence['language']] for sentence in samples])}
