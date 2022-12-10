import json
from tqdm import tqdm
import requests
from nltk import tokenize

##################################################################
# reference: https://huggingface.co/docs/api-inference/quicktour #
##################################################################


API_TOKEN = "" # YOU SHOULD GET THIS BY LOGIN HUGGINGFACE
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ar-en"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if len(response.json()) != 1:
        print('')
        print(response.json())
        raise "Reponse error!"
    return response.json()

if __name__ == '__main__':
    start_idx = 1600
    dataset_size = 2

    with open("arabic.json", "r", encoding='utf8') as file:
        all_paragraphs = json.load(file)
    selected_paragraphs = all_paragraphs[start_idx:start_idx+dataset_size]

    translated_paragraphs = []
    for i in tqdm(range(dataset_size)):
        sentences = tokenize.sent_tokenize(selected_paragraphs[i]['paragraph'])
        tmp = []
        for sentence in sentences:
            if len(sentence) > 512: # max length for huggingface models
                continue
            tmp.append(
                query({ "inputs": sentence, })[0]["translation_text"]
            )
        
        translated_paragraphs.append({
            'paragraph': " ".join(tmp),
            'language': 'arabic'
        })

    to_dump = json.dumps(translated_paragraphs, indent=4, ensure_ascii=False)
    with open("train_opus_translate_0_200.json", "w", encoding='utf8') as file:
        file.write(to_dump)
