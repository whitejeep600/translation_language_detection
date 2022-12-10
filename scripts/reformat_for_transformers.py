import json
from pathlib import Path

if __name__ == '__main__':
    all_paragraphs = json.loads(Path('data/original/indonesian/all.json').read_text())
    #selected_paragraphs = all_paragraphs[1600:3550]  # 1600 + 1600 for train + 350 for test - for monolingual
    selected_paragraphs = all_paragraphs[3550:3900]  # 350 paragraphs for test
    reformatted = [{"translation":
                        {"in": paragraph['paragraph'],
                         "en": "lolita, light of my life, fire of my loins"}}
                   for paragraph in selected_paragraphs]
    with open('data/original/indonesian/for_multilingual.json', 'w', encoding='utf8') as file:
        for paragraph in reformatted:
            file.write(json.dumps(paragraph))
            file.write("\n")
