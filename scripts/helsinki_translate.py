import json

from transformers import pipeline

if __name__ == '__main__':
    model_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    translator = pipeline("translation", model=model_checkpoint)
    paragraphs = []
    with open('data/original/indonesian/for_monolingual.jsonl') as f:
        for line in f:
            paragraphs.append(json.loads(line))

    train_paragraphs = []
    for i in range(1600):
        train_paragraphs.append({'paragraph': translator(paragraphs[i])[0]['translation_text'],
                                 'language': 'indonesian'})
    train_json = json.dumps(train_paragraphs, indent=4)
    with open("../data/translated/from_indonesian/train_helsinki.json", "w") as file:
        file.write(train_json)

    test_paragraphs = []
    for i in range(1600, 1950):
        test_paragraphs.append({'paragraph': translator(paragraphs[i])[0]['translation_text'],
                                'language': 'indonesian'})
    test_json = json.dumps(test_paragraphs, indent=4)
    with open("../data/translated/from_indonesian/test_helsinki.json", "w") as file:
        file.write(test_json)
