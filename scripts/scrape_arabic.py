import json
import requests
from bs4 import BeautifulSoup
from re import sub, fullmatch
from nltk import tokenize
from tqdm import tqdm
import random


def get_chunks_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    text = ' '.join([data.get_text() for data in parsed_response.find(id='bodyContent').find_all('p')])
    
    # removing footnotes - they are particularly frequent in Wikipedia articles
    text = sub(r'\[[0-9\]]*', '', text)
    text = sub(r'\n', '', text)
    text = sub(r'\t', '', text)
    
    # filter the text
    if 'صفحات للمحرري' in text or '1٬196٬094 مقالة بالعرب' in text: # if 'Pages for editors' or '1,196,094 articles in the Arabs' in text
        return []
    if 'عن ويكيبيد' in text or 'كيفية تعديل الصفحات' in text: # if 'About Wikipedia' or 'How to edit pages' in text
        return []
    
    sentences = tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # filter the sentence
        if 'ويكيبيديا' in sentence: # if 'wikipedia' in sentence
            continue
        
        if len(current_chunk.split()) + len(sentence.split()) < 256:
            current_chunk += sentence + ' '
        else:
            current_chunk += sentence + ' '
            chunks.append(current_chunk)
            current_chunk = ''
    return chunks

MULTIMEDIA_FORMAT = {'png', 'jpg', 'gif', 'mid', 'midi', 'ogg'}
def get_links_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    link_objects = parsed_response.find(id='bodyContent').find_all('a')
    bare_links = [link.get('href') for link in link_objects
                  if link.get('href') and link.get('href').split('.')[-1].lower() not in MULTIMEDIA_FORMAT]
    
    bare_links = random.sample(bare_links, len(bare_links) // 5) # limit the breadth of the BFS by random sample
    return ['https://ar.wikipedia.org' + link for link in bare_links]

def main():
    all_chunks = []
    visited_sites = set()
    reachable_sites = set()
    reachable_sites.add('https://ar.wikipedia.org/wiki/المجموعة_الشمسية')
    
    data_size = 7500
    progress = tqdm(total=data_size) # add progress bar
    while len(all_chunks) < data_size:
        try:
            http_response = requests.get(url=reachable_sites.pop())
        except requests.ConnectionError:
            continue
        
        if len(reachable_sites) < 1000: # constrain the size of BFS container
            new_reachable_sites = get_links_from_response(http_response)
            for site in new_reachable_sites:
                if site not in visited_sites:
                    visited_sites.add(site)
                    reachable_sites.add(site)
        for chunk in get_chunks_from_response(http_response):
            all_chunks.append(chunk)
            progress.update(1)
    
    random.shuffle(all_chunks) # shuffle the list
    final_json = json.dumps([{'paragraph': paragraph.rstrip(), 'language': 'arabic'} for paragraph in all_chunks], indent=4, ensure_ascii=False)
    with open("arabic.json", "w", encoding='utf8') as file:
        file.write(final_json)


if __name__ == '__main__':
    main()

