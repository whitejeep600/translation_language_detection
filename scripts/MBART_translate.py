import json

from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_paragraph(paragraph, tokenizer, model):
    encoded = tokenizer(paragraph, return_tensors="pt", max_length=512, truncation=True).to('cuda:0')
    generated_tokens = model.generate(**encoded, max_length=512)
    res =  tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, max_length=512)
    print(res)
    return res


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
    model_name = 'facebook/mbart-large-50-many-to-one-mmt'
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = "id_ID"
    model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda:0')

    translated_paragraphs = translate_paragraphs(paragraphs, tokenizer, model)
    train_paragraphs = translated_paragraphs[:1600]
    test_paragraphs = translated_paragraphs[1600:]

    test_json = json.dumps(test_paragraphs, indent=4)
    with open("data/translated/from_indonesian/test_MBART.jsonl", "w") as file:
        file.write(test_json)

    train_json = json.dumps(train_paragraphs, indent=4)
    with open("data/translated/from_indonesian/train_MBART.jsonl", "w") as file:
        file.write(train_json)
