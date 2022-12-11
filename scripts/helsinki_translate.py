import json

from tqdm import tqdm
from transformers import pipeline


def translate(paragraphs, indices, translator):
    translated = []
    skipped = 0
    progress = tqdm(total=len(indices))
    for i in indices:
        if len(translator.tokenizer.tokenize(paragraphs[i]['translation']['in'])) < 512:
            translated.append(
                {'paragraph': translator(paragraphs[i]['translation']['in'], max_length=512)[0]['translation_text'],
                 'language': 'indonesian'})
        else:
            skipped += 1
        progress.update(1)
    print(f'skipped {skipped} paragraphs which exceeded max lengthZ')
    return translated


if __name__ == '__main__':
    model_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    translator = pipeline("translation", model=model_checkpoint, device=0)
    paragraphs = []
    with open('data/original/indonesian/for_monolingual.jsonl') as f:
        for line in f:
            paragraphs.append(json.loads(line))

    train_paragraphs = translate(paragraphs,range(1600, 3200), translator)
    train_json = json.dumps(train_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_helsinki.json", "w") as file:
        file.write(train_json)

    test_paragraphs = translate(paragraphs, range(3200, 3550), translator)
    test_json = json.dumps(test_paragraphs, indent=4)
    with open("data/translated/from_indonesian/test_helsinki.json", "w") as file:
        file.write(test_json)
