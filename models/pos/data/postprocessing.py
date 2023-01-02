import json
from tqdm import tqdm

def replace_words():
    with open("data/arabic.json", "r", encoding='utf8') as file:
        all_paragraphs = json.load(file)
    
    for i in tqdm(range(len(all_paragraphs))):
        all_paragraphs[i]["paragraph"] = all_paragraphs[i]["paragraph"].replace("تعديل - تعديل مصدري - تعديل ويكي بيانات", "").strip()
    to_dump = json.dumps(all_paragraphs, indent=4, ensure_ascii=False)
    with open("data/arabic.json", "w", encoding='utf8') as file:
        file.write(to_dump)

def remove_paragraphs(remove_idx):
    with open("data/arabic.json", "r", encoding='utf8') as file:
        all_paragraphs = json.load(file)
    
    res = []
    for i in tqdm(range(len(all_paragraphs))):
        if i in remove_idx:
            continue
        res.append(all_paragraphs[i])

    to_dump = json.dumps(all_paragraphs, indent=4, ensure_ascii=False)
    with open("data/arabic_revised.json", "w", encoding='utf8') as file:
        file.write(to_dump)

def merge_data(file_list, name):
    paragraphs_list = []
    for file in file_list:
        with open(file, "r", encoding='utf8') as file:
            data = json.load(file)
            print(len(data))
            paragraphs_list += data
    to_dump = json.dumps(paragraphs_list, indent=4, ensure_ascii=False)
    with open(name, "w", encoding='utf8') as file:
        file.write(to_dump)

def remove_duplicated():
    with open("data/arabic_revised.json", "r", encoding='utf8') as file:
        data = json.load(file)
    with open("data/translated/train_arabic_raw.json", "r", encoding='utf8') as file:
        translated = json.load(file)
    
    chk = set()
    duplicated = set()
    for i in range(7519):
        if data[i]['paragraph'] in chk:
            duplicated.add(i)
        else:
            chk.add(data[i]['paragraph'])
    
    res = []
    for i in tqdm(range(len(translated))):
        if i in duplicated:
            continue
        
        res.append(translated[i])
        if len(res) > 6400:
            if i < 6897:
                res[-1]['translator'] = "facebook/mbart-large-50-many-to-one-mmt"
            else:
                res[-1]['translator'] = "Helsinki-NLP/opus-mt-ar-en"
    print(len(res))
    print(len(translated))
    print(len(duplicated))
    to_dump = json.dumps(res, indent=4, ensure_ascii=False)
    with open("data/translated/train_arabic_raw.json", "w", encoding='utf8') as file:
        file.write(to_dump)

def parsing_dataset():
    with open("data/translated/train_arabic_raw.json", "r", encoding='utf8') as file:
        translated = json.load(file)

    # chk = set()
    # for i in range(len(translated)):
        # para_len = len(translated[i]['paragraph'].split())
        # if para_len > 300:
            # chk.add(i)
    
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    
    # from nltk import tokenize
    # sentences = tokenize.sent_tokenize(translated[0]['paragraph'])
    # for s in sentences:
        # print(len(s))
    # print(len(translated[0]['paragraph'].split()))
    
    train = []
    test = []
    i = 0
    for i in range(len(translated)):
        if len(translated[i]['paragraph'].split()) < 240:
            continue
    
        if len(train) == 6400:
            test.append(translated[i])
            if len(test) == 700:
                break
            continue
        train.append(translated[i])
    
    print(len(train))
    print(len(test))
    
    train_dump = json.dumps(train, indent=4, ensure_ascii=False)
    test_dump = json.dumps(test, indent=4, ensure_ascii=False)
    with open("data/translated/train_arabic.json", "w", encoding='utf8') as file:
        file.write(train_dump)
    with open("data/translated/test_arabic.json", "w", encoding='utf8') as file:
        file.write(test_dump)

def remove_test():
    with open("test_pos_raw.json", "r", encoding='utf8') as file:
        translated = json.load(file)
    
    res = []
    chinese = {'mBART': [], 'helsinki': []}
    for item in translated:
        if item['language'] == 'chinese':
            chinese[item['translator']].append(item)
        else:
            res.append(item)
    
    res.extend(chinese['mBART'][:350])
    res.extend(chinese['helsinki'][:350])
    
    test_dump = json.dumps(res, indent=4, ensure_ascii=False)
    with open("test_pos.json", "w", encoding='utf8') as file:
        file.write(test_dump)
    


if __name__ == '__main__':
    # lang = ['arabic', 'chinese', 'indonesian', 'japanese']
    # train_files = [f'translated/from_{lang[i]}/train_{lang[i]}.json' for i in range(4)]
    # merge_data(train_files, 'train.json')
    # test_files = [f'translated/from_{lang[i]}/test_{lang[i]}.json' for i in range(4)]
    # merge_data(test_files, 'test.json')
    
    
    # merge_data(['train_pos1.json', 'train_pos2.json'], 'train_pos.json')
    remove_test()
    pass