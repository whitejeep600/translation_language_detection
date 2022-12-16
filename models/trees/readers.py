import json
from pathlib import Path

from nltk import tokenize


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

