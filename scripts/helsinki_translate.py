import json

from nltk import tokenize
from tqdm import tqdm
from transformers import pipeline


def translate_sentence(sentence, translator):
    if len(translator.tokenizer.tokenize(sentence)) < 256:
        return translator(sentence, max_length=256)[0]['translation_text']
    else:
        return ""


def translate_text(text, translator):
    sentences = tokenize.sent_tokenize(text)
    result = ' '.join([translate_sentence(sentence, translator)
                      for sentence in sentences])
    assert result  # making sure we didn't skip ALL sentences
    return result


def translate_paragraph(paragraphs, indices, translator):
    translated = []
    progress = tqdm(total=len(indices))
    for i in indices:
        translated.append(
                {'paragraph': translate_text(paragraphs[i]['translation']['in'], translator),
                 'language': 'indonesian'})
        progress.update(1)
    return translated


if __name__ == '__main__':
    model_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    translator = pipeline("translation", model=model_checkpoint, device=0)
    paragraphs = []
    with open('data/original/indonesian/for_monolingual.jsonl') as f:
        for line in f:
            paragraphs.append(json.loads(line))

    test_paragraphs = translate_paragraph(paragraphs, range(1600, 1950), translator)
    test_json = json.dumps(test_paragraphs, indent=4)
    with open("data/translated/from_indonesian/test_helsinki.json", "w") as file:
        file.write(test_json)

    train_paragraphs = translate_paragraph(paragraphs, range(1600), translator)
    train_json = json.dumps(train_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_helsinki.json", "w") as file:
        file.write(train_json)


