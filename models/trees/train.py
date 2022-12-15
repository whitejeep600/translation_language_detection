import json
from pathlib import Path

from nltk import tokenize


def read_train_dataset():
    path = '../../data/translated/from_indonesian/train_indonesian.json'
    all_paragraphs = json.loads(Path(path).read_text(encoding='utf-8'))
    split_paragraphs = [tokenize.sent_tokenize(paragraph['paragraph']) for paragraph in all_paragraphs]
    return split_paragraphs


if __name__ == '__main__':
    all_sentences = read_train_dataset()
    pass
