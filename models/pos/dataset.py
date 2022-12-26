from typing import List, Dict
from torch.utils.data import Dataset
import torch
import numpy as np

vocab = {
    "NOUN": 0,
    "PUNCT": 1,
    "ADP": 2,
    "DET": 3,
    "PROPN": 4,
    "VERB": 5,
    "ADJ": 6,
    "AUX": 7,
    "CCONJ": 8,
    "PRON": 9,
    "NUM": 10,
    "ADV": 11,
    "PART": 12,
    "SCONJ": 13,
    "SYM": 14,
    "INTJ": 15,
    "X": 16,
    "PAD": 17,
}
def encode_paragraph(paragraph):
    texts = paragraph.split()
    texts = texts[:256] + ["PAD"] * max(0, 256 - len(texts))
    
    encoded = [vocab[t] for t in texts]
    ret = [[0]*18 for i in range(256)]
    for i in range(256):
        ret[i][encoded[i]] = 1
    return ret

def encode_translator(s):
    if s == 'facebook/mbart-large-50-many-to-one-mmt':
        return 'mbart'
    elif s == 'Helsinki-NLP/opus-mt-ar-en':
        return 'helsinki'
    elif s == 'staka':
        return 'helsinki'
    else:
        return s.lower()

lang = {
    "arabic": 0,
    "chinese": 1,
    "indonesian": 2,
    "japanese": 3,
}

class TranslationDataset(Dataset):
    def __init__(
        self,
        data: List[Dict]
    ):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:
        batch = {}
        batch['input'] = torch.tensor([encode_paragraph(s['paragraph']) for s in samples], dtype=torch.float32)
        batch['gt'] = torch.tensor([lang[s['language']] for s in samples], dtype=torch.long)
        
        if 'translator' in samples[0].keys():
            batch['translator'] = [encode_translator(s['translator']) for s in samples]
        
        return batch


if __name__ == '__main__':
    p = "NOUN VERB ADJ"
    r = encode_paragraph(p)
    print(r[:3])