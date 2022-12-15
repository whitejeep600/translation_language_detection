import json
from pathlib import Path

if __name__ == '__main__':
    path = 'data/translated/from_indonesian/test_indonesian.json'
    all_paragraphs = json.loads(Path(path).read_text(encoding='utf-8'))
    for paragraph in all_paragraphs:
        if not isinstance(paragraph['paragraph'], str):
            paragraph['paragraph'] = paragraph['paragraph'][0]
    train_dump = json.dumps(all_paragraphs, indent=4)
    with open('data/translated/from_indonesian/test_indonesian2.json', 'a') as f:
        f.write(train_dump)
