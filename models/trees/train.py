import json
import random
from pathlib import Path

import torch
from nltk import tokenize
from torch.utils.data import DataLoader

from translation_language_detection.models.trees.constants import label_to_int
from translation_language_detection.models.trees.dataset import TranslationDetectionDataset


def read_language(language):
    path = '../../data/translated/from_' + language + '/train_' + language + '.json'
    all_paragraphs = json.loads(Path(path).read_text(encoding='utf-8'))
    sentences = []
    for paragraph in all_paragraphs:
        for sentence in tokenize.sent_tokenize(paragraph['paragraph']):
            if len(sentence) > 8:  # shorter ones are likely parse errors
                sentences += [{'text': sentence, 'language': language}]
    return sentences


def read_training_data():
    languages = ['indonesian', 'arabic']
    data_by_language = [read_language(language) for language in languages]
    all_data = []
    for data in data_by_language:
        all_data += data
    return all_data


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
