import json

import torch
from nltk import tokenize

from constants import SAVE_DIR, MAX_SENTENCE_LENGTH, NUM_LABELS, D, LANGUAGES, TEST_TRANSLATORS
from model import TranslationDetector
from dependency_parsing import sentence_to_matrix
from readers import read_data


class Tester:
    def __init__(self, model, paragraphs):
        self.model = model
        self.paragraphs = paragraphs

    def correct(self, paragraph):
        text_tensor = torch.stack([sentence_to_matrix(sentence)
                                   for sentence in tokenize.sent_tokenize(paragraph['paragraph'])])
        predictions_tensor = self.model(text_tensor)
        return torch.mode(predictions_tensor) == paragraph['language']

    def test(self):
        correct = 0
        total = len(self.paragraphs)
        result = {language:
                      {translator: {
                          'correct': 0,
                          'incorrect': 0
                      }
                       for translator in TEST_TRANSLATORS}
                  for language in LANGUAGES}
        result_json = json.dumps(result, indent=4)
        for paragraph in self.paragraphs:
            if self.correct(paragraph):
                result[paragraph['language']][paragraph['translator']]['correct'] += 1
            else:
                result[paragraph['language']][paragraph['translator']]['incorrect'] += 1
        with open("test_results.json", "w") as file:
            file.write(result_json)
        print(f'Correct: {correct} out of {total}.')


def load_model():
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"  # always using GPU if available
    model = TranslationDetector(D, MAX_SENTENCE_LENGTH, NUM_LABELS)
    model.load_state_dict(torch.load(SAVE_DIR))
    model.eval()
    model.to(target_device)
    return model


if __name__ == '__main__':
    tester = Tester(load_model(), read_data(testing=True)[10])
    tester.test()
