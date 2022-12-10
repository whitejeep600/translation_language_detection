import json
import random

import requests
from bs4 import BeautifulSoup
from re import sub, fullmatch
from nltk import tokenize
from tqdm import tqdm


def get_chunks_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    content = parsed_response.find(id='bodyContent')
    if content == None:
        return []
    text = ' '.join([data.get_text() for data in content.find_all('p')])
    text = sub(r'\[[0-9\]]*', '', text)
    text = sub(r'\n', '', text)
    text = sub(r'\t', '', text)
    if 'Tidak ada teks di halaman ini' in text or 'Tentang Wikipedia' in text or \
        'Tidak ditemukan hasil untuk' in text:
        return []
    # removing footnotes - they are particularly frequent in Wikipedia articles
    sentences = tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if 'Wikipedia' in sentence:
            continue
        if len(current_chunk.split()) + len(sentence.split()) < 256:
            current_chunk += sentence + ' '
        else:
            current_chunk += sentence + ' '
            chunks.append(current_chunk)
            current_chunk = ''
    return chunks


def get_links_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    link_objects = parsed_response.find(id='bodyContent').find_all('a')
    bare_links = [link.get('href') for link in link_objects
                  if link.get('href') and fullmatch(r"/wiki/[a-zA-Z]*", link.get('href'))]
    # here I selected the alphanumeric links to avoid stupid links to .png
    # images etc. Will need to be adjusted, or I think it can be removed
    # for non-latin-alphabet languages.
    bare_links = random.sample(bare_links, len(bare_links) // 5)
    return ['https://id.wikipedia.org' + link for link in bare_links]
    # replace 'id' with the prefix of your language


if __name__ == '__main__':
    all_chunks = []
    visited_sites = set()
    reachable_sites = set()
    reachable_sites.add('https://id.wikipedia.org/wiki/Filsafat')
    # replace the address with an article in your language
    data_size = 9000
    progress = tqdm(total=data_size)  # add progress bar

    while len(all_chunks) < data_size:
        try:
            http_response = requests.get(url=reachable_sites.pop())
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

    final_json = json.dumps([{'paragraph': paragraph, 'language': 'indonesian'} for paragraph in all_chunks], indent=4)
    with open("../data/original/indonesian/indonesian.json", "w") as file:
        file.write(final_json)
    # of course replace 'indonesian' here