import random

import torch
from torch.utils.data import DataLoader

from translation_language_detection.models.trees.constants import label_to_int
from translation_language_detection.models.trees.dataset import TranslationDetectionDataset
from translation_language_detection.models.trees.readers import read_training_data


class Trainer:
    pass


def create_dataloader(split):
    dataset = TranslationDetectionDataset(split, label_to_int)
    return DataLoader(dataset, batch_size=32, shuffle=True,
                      collate_fn=TranslationDetectionDataset.collate_fn)


if __name__ == '__main__':
    all_sentences = read_training_data()
    random.shuffle(all_sentences)
    validation_split = all_sentences[:len(all_sentences) // 10]
    train_split = all_sentences[len(all_sentences) // 10:]
    validation_loader = create_dataloader(validation_split)
    train_loader = create_dataloader(train_split)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"  # always using GPU if available
    pass
