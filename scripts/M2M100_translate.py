import json

from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def translate_paragraph(paragraph, tokenizer, model):
    encoded = tokenizer(paragraph, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en"))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate_paragraphs(paragraphs, tokenizer, model):
    translated = []
    progress = tqdm(total=len(paragraphs))
    for paragraph in paragraphs:
        translated.append(
            {'paragraph': translate_paragraph(paragraph['translation']['in'], tokenizer, model),
             'language': 'indonesian'})
        progress.update(1)
    return translated


if __name__ == '__main__':
    paragraphs = []
    with open('data/original/indonesian/for_multilingual.jsonl') as f:
        for line in f:
            paragraphs.append(json.loads(line))

    assert(len(paragraphs) == 1600 + 350)

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="id")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    translated_paragraphs = translate_paragraphs(paragraphs, tokenizer, model)
    train_paragraphs = translated_paragraphs[:1600]
    test_paragraphs = translated_paragraphs[1600:]

    test_json = json.dumps(test_paragraphs, indent=4)
    with open("data/translated/from_indonesian/test_M2M100.json", "w") as file:
        file.write(test_json)

    train_json = json.dumps(train_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_M2M100.json", "w") as file:
        file.write(test_json)
