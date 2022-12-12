import json
import random

import requests
from bs4 import BeautifulSoup
from re import sub, fullmatch
from nltk import tokenize
from tqdm import tqdm
from konoha import SentenceTokenizer
from janome.tokenizer import Tokenizer

sent_tokenizer = SentenceTokenizer()
word_tokenizer = Tokenizer()
def get_chunks_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    content = parsed_response.find(id='bodyContent')
    if content == None:
        return []
    text = ' '.join([data.get_text() for data in content.find_all('p')])
    text = sub(r'\[[0-9\]]*', '', text)
    text = sub(r'\n', '', text)
    text = sub(r'\t', '', text)
    if "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30" in content:
        return []
    # removing footnotes - they are particularly frequent in Wikipedia articles
    sentences = sent_tokenizer.tokenize(text)
    chunks = []
    current_chunk = ""
    current_size = 0
    for sentence in sentences:
        words = word_tokenizer.tokenize(sentence, wakati=True)
        count = 0
        for w in words:
            count += 1
        if 'Wikipedia' in sentence:
            continue
        if current_size + count < 256:
            current_chunk += sentence + ' '
            current_size += count
        else:
            current_chunk += sentence + ' '
            chunks.append(current_chunk)
            if(len(chunks) > 1):
                break
            current_chunk = ''
            current_size = 0
            
    return chunks


MULTIMEDIA_FORMAT = {'png', 'jpg', 'gif', 'mid', 'midi', 'ogg'}
def get_links_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    link_objects = parsed_response.find(id='bodyContent').find_all('a')
    bare_links = [link.get('href') for link in link_objects
                  if link.get('href') and link.get('href').split('.')[-1].lower() not in MULTIMEDIA_FORMAT]
    bare_links = random.sample(bare_links, len(bare_links) // 5)
    return ['https://ja.wikipedia.org' + link for link in bare_links]
    # replace 'id' with the prefix of your language


if __name__ == '__main__':
    all_chunks = []
    visited_sites = set()
    reachable_sites = set()
    reachable_sites.add('https://ja.wikipedia.org/wiki/孔子')
    # replace the address with an article in your language
    data_size = 7100
    progress = tqdm(total=data_size)  # add progress bar

    while len(all_chunks) < data_size:
        try:
            target_site = reachable_sites.pop()
            http_response = requests.get(url=target_site)
        except requests.ConnectionError:
            continue

        if len(reachable_sites) < 1000:  # constrain the size of BFS container
            new_reachable_sites = get_links_from_response(http_response)
            for site in new_reachable_sites:
                if site not in visited_sites:
                    visited_sites.add(site)
                    reachable_sites.add(site)
        for chunk in get_chunks_from_response(http_response):
            all_chunks.append(chunk)
            progress.update(1)

    final_json = json.dumps([{'paragraph': paragraph, 'language': 'japanese'} for paragraph in all_chunks], indent=4)

    with open("../../data/original/japanese/japanese3.json", "w") as file:
        file.write(final_json)
    # of course replace 'indonesian' here