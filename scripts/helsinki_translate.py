import json

from tqdm import tqdm
from transformers import pipeline

if __name__ == '__main__':
    model_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    translator = pipeline("translation", model=model_checkpoint)
    paragraphs = []
    with open('data/original/indonesian/for_monolingual.jsonl') as f:
        for line in f:
            paragraphs.append(json.loads(line))

    train_paragraphs = []
    progress = tqdm(total=1600)
    for i in range(1600, 3200):
        train_paragraphs.append({'paragraph': translator(paragraphs[i]['translation']['in'], max_length=1024)[0]['translation_text'],
                                 'language': 'indonesian'})
        progress.update(1)
    train_json = json.dumps(train_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_helsinki.json", "w") as file:
        file.write(train_json)

    test_paragraphs = []
    progress = tqdm(total=350)
    for i in range(3200, 3550):
    	test_paragraphs.append({'paragraph': translator(paragraphs[i]['translation']['in'], max_length=1024)[0]['translation_text'],
                                'language': 'indonesian'})
        progress.update(1)
    test_json = json.dumps(test_paragraphs, indent=4)
    with open("data/translated/from_indonesian/test_helsinki.json", "w") as file:
        file.write(test_json)
