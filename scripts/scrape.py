import json

import requests
from bs4 import BeautifulSoup
from re import sub, fullmatch
from nltk.tag import pos_tag  # execute: nltk.download('averaged_perceptron_tagger')
from nltk import tokenize


def contains_proper_name(sentence):
    sentence = pos_tag(sentence.split())
    res = any([pos == 'NNP' for _, pos in sentence])
    return res


def remove_proper_names(sentence):
    split_sentence = pos_tag(sentence.split())
    proper_nouns = [word for word, pos in split_sentence if pos == 'NNP']
    for noun in proper_nouns:
        sentence = sentence.replace(noun, '')
    return sentence


def get_chunks_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    text = ' '.join([data.get_text() for data in parsed_response.find_all('p')])
    text = sub(r'\[[0-9\]]*', '', text)
    text = sub(r'\n', '', text)
    text = sub(r'\t', '', text)
    # removing footnotes - they are particularly frequent in Wikipedia articles
    sentences = tokenize.sent_tokenize(text)
    sentences = [remove_proper_names(sentence) for sentence in sentences]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < 256:
            current_chunk += sentence + ' '
        else:
            current_chunk += sentence + ' '
            chunks.append(current_chunk)
            current_chunk = ''
    return chunks


def get_links_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    links = parsed_response.find(id='bodyContent').find_all('a')
    bare_links = [link.get('href') for link in links
                  if link.get('href') and fullmatch(r"/wiki/[a-zA-Z]*", link.get('href'))]
    return ['https://id.wikipedia.org' + link for link in bare_links]
    # replace 'id' with the prefix of your language


if __name__ == '__main__':
    all_chunks = []
    visited_sites = set()
    reachable_sites = set()
    reachable_sites.add('https://id.wikipedia.org/wiki/Filsafat')
    # replace the address with an article in your language
    while len(all_chunks) < 7000:
        http_response = requests.get(url=reachable_sites.pop())
        new_reachable_sites = get_links_from_response(http_response)
        for site in new_reachable_sites:
            if site not in visited_sites:
                visited_sites.add(site)
                reachable_sites.add(site)
        for chunk in get_chunks_from_response(http_response):
            all_chunks.append(chunk)
    final_json = json.dumps([{'paragraph': paragraph, 'language': 'indonesian'} for paragraph in all_chunks], indent=4)
    with open("indonesian.json", "w") as file:
        file.write(final_json)
    # of course replace 'indonesian' here

