from typing import List, Dict

from torch.utils.data import Dataset


class TranslationDetectionDataset(Dataset):
    def __init__(self, data: List[Dict], label_mapping: Dict[str, int],):
        self.data = data
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]
