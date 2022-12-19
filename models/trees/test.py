import json

import torch
from nltk import tokenize
from tqdm import tqdm

from constants import SAVE_DIR, MAX_SENTENCE_LENGTH, NUM_LABELS, D, LANGUAGES, TEST_TRANSLATORS, LABEL_TO_INT
from model import TranslationDetector
from dependency_parsing import sentence_to_matrix
from readers import read_data
from utils import get_target_device


class Tester:
    def __init__(self, model, paragraphs):
        self.model = model
        self.paragraphs = paragraphs

    def correct(self, paragraph):
        text_tensor = torch.stack([sentence_to_matrix(sentence)
                                   for sentence in tokenize.sent_tokenize(paragraph['paragraph'])])
        predictions_tensor = self.model(text_tensor)
        return torch.mode(torch.argmax(predictions_tensor, dim=1))[0].item() == LABEL_TO_INT[paragraph['language']]

    def test(self):
        correct = 0
        total = len(self.paragraphs)
        result = {language:
                      {translator: {
                          'correct': 0,
                          'incorrect': 0
                      }
                       for translator in TEST_TRANSLATORS[language]}
                  for language in LANGUAGES}
        progress = tqdm(total=len(self.paragraphs), desc="Processed paragraph")
        for paragraph in self.paragraphs:
            if self.correct(paragraph):
                result[paragraph['language']][paragraph['translator']]['correct'] += 1
                correct += 1
            else:
                result[paragraph['language']][paragraph['translator']]['incorrect'] += 1
            progress.update(1)
        result_json = json.dumps(result, indent=4)
        with open("test_results.json", "w") as file:
            file.write(result_json)
        print(f'Correct: {correct} out of {total}.')


def load_model():
    model = TranslationDetector(D, MAX_SENTENCE_LENGTH, NUM_LABELS)
    model.load_state_dict(torch.load(SAVE_DIR))
    model.eval()
    model.to(get_target_device())
    return model


if __name__ == '__main__':
    tester = Tester(load_model(), read_data(testing=True))
    tester.test()
