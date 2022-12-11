import json

from tqdm import tqdm
from transformers import pipeline

if __name__ == '__main__':
    model_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    translator = pipeline("translation", model=model_checkpoint, device=0)
    paragraphs = []
    with open('data/original/indonesian/for_monolingual.jsonl') as f:
        for line in f:
            paragraphs.append(json.loads(line))

    train_paragraphs = []
    skipped_train = 0
    progress = tqdm(total=1600)
    for i in range(1600, 3200):
        if len(translator.tokenizer.tokenize(paragraphs[i]['translation']['in'])) < 512:
            train_paragraphs.append({'paragraph': translator(paragraphs[i]['translation']['in'], max_length=512)[0]['translation_text'],
                                     'language': 'indonesian'})
        else:
            skipped_train += 1
        progress.update(1)
    train_json = json.dumps(train_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_helsinki.json", "w") as file:
        file.write(train_json)

    test_paragraphs = []
    progress = tqdm(total=350)
    for i in range(3200, 3550):
        test_paragraphs.append({'paragraph': translator(paragraphs[i]['translation']['in'], max_length=512)[0]['translation_text'],
                                'language': 'indonesian'})
        progress.update(1)
    test_json = json.dumps(test_paragraphs, indent=4)
    with open("data/translated/from_indonesian/test_helsinki.json", "w") as file:
        file.write(test_json)
