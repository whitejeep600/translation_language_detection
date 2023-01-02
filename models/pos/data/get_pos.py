import json

from nltk import tokenize
from tqdm import tqdm
from transformers import pipeline


def get_sentence_pos(sentence, token_classifier):
    return token_classifier(sentence)


def get_text_pos(text, token_classifier):
    sentences = tokenize.sent_tokenize(text)
    res = []
    for sentence in sentences:
        res += [s['entity_group'] for s in get_sentence_pos(sentence, token_classifier)]
    return ' '.join(res)

def main():
    token_classifier = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", device=0, aggregation_strategy="simple")
    
    name = 'test'
    with open(f"{name}.json", "r", encoding='utf8') as file:
        all_paragraphs = json.load(file)
    
    japanese = []
    bad = []
    paragraphs_pos = []
    for i in tqdm(range(len(all_paragraphs))):
        try:
            result = get_text_pos(all_paragraphs[i]['paragraph'], token_classifier)
        except:
            japanese.append(i)
            continue
    
        
        if len(result) == 0:
            print(i)
            bad.append(i)
            continue
        if name == 'test':
            paragraphs_pos.append(
                    {'paragraph': result,
                     'language': all_paragraphs[i]['language'],
                     'translator': all_paragraphs[i]['translator']
                    })
        else:
            paragraphs_pos.append(
                    {'paragraph': result,
                     'language': all_paragraphs[i]['language']})
        
        if (i+1) % 1000 == 0:
            to_dump = json.dumps(paragraphs_pos, indent=4, ensure_ascii=False)
            with open(f"{name}_pos.json", "w", encoding='utf8') as file:
                file.write(to_dump)
    
    print(japanese)
    print('***')
    print(bad)
    to_dump = json.dumps(paragraphs_pos, indent=4, ensure_ascii=False)
    with open(f"{name}_pos.json", "w", encoding='utf8') as file:
        file.write(to_dump)

def check():
    print(len(japanese))
    with open("train.json", "r", encoding='utf8') as file:
        all_paragraphs = json.load(file)
    
    # print(len(bad))
    # print(all_paragraphs[18919])
    # print(all_paragraphs[18920])
    # for i in range(26,len(bad)):
        # print(all_paragraphs[bad[i]])
    
    test = 123
    print(all_paragraphs[test])
    token_classifier = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", device=0, aggregation_strategy="simple")
    result = get_text_pos(all_paragraphs[test], token_classifier)

def test():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, padding=False, max_length=256, truncation=True)
    print(encoded_input)
    # output = model(**encoded_input)

if __name__ == '__main__':
    main()
    # check()
    
    # test()