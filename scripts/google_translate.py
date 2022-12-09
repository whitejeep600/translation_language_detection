import json
from pathlib import Path

from googletrans import Translator

if __name__ == '__main__':
    all_paragraphs = json.loads(Path('data/original/indonesian.json').read_text())
    selected_paragraphs = all_paragraphs[:1600]
    t = Translator()
    translated_paragraphs = [{'paragraph': t.translate(paragraph['paragraph']).text,
                              'language': 'indonesian'} for paragraph in selected_paragraphs]
    to_dump = json.dumps(translated_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_google_translate.json", "w") as file:
        file.write(to_dump)
