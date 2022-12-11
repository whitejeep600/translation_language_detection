import json
from pathlib import Path


def reformat_and_save_paragraphs(paragraphs, filename):
    reformatted = [{"translation":
                        {"in": paragraph['paragraph'],
                         "en": "lolita, light of my life, fire of my loins"}}
                   # format required for huggingface example scripts (eventually not used)
                   for paragraph in paragraphs]
    with open(filename, 'w', encoding='utf8') as file:
        for paragraph in reformatted:
            file.write(json.dumps(paragraph))
            file.write("\n")


if __name__ == '__main__':
    all_paragraphs = json.loads(Path('data/original/indonesian/all.jsonl').read_text())
    # the first are 1600 for the Google Translate API
    paragraphs_for_mono = all_paragraphs[1600:3550]  # 1600 train + 350 test for a monolingual model
    reformat_and_save_paragraphs(paragraphs_for_mono, 'data/original/indonesian/for_monolingual.jsonl')

    paragraphs_for_multi = all_paragraphs[3550:5500]  # 1600 train + 350 test for a monolingual model
    reformat_and_save_paragraphs(paragraphs_for_multi, 'data/original/indonesian/for_multilingual.jsonl')

    paragraphs_for_api = all_paragraphs[5500:7100]
    reformat_and_save_paragraphs(paragraphs_for_api, 'data/original/indonesian/for_api.jsonl')
