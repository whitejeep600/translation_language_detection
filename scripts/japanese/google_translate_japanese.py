import json
import httpcore
from pathlib import Path
import time
from googletrans import Translator

if __name__ == '__main__':
    all_paragraphs = json.loads(Path('../../data/original/japanese/japanese3.json').read_text())
    selected_paragraphs = all_paragraphs[:1600]
    t = Translator()
    translated_paragraphs = []
    for paragraph in selected_paragraphs:
        print('--------------------')
        print(paragraph)
        try:
            translated_paragraphs.append({'paragraph': t.translate(paragraph['paragraph']).text,
                              'language': 'japanese'})
        except TypeError or AttributeError or httpcore._exceptions.ReadTimeout:
            to_dump = json.dumps(translated_paragraphs, indent=4)
            with open("../../data/translated/from_japanese/train_google_translate.json", "w") as file:
                file.write(to_dump)
        time.sleep(0.5)
    to_dump = json.dumps(translated_paragraphs, indent=4)
    with open("../../data/translated/from_japanese/train_google_translate.json", "w") as file:
        file.write(to_dump)
