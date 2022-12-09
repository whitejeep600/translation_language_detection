import json
import requests
from bs4 import BeautifulSoup
from re import sub, fullmatch
from nltk import tokenize
from tqdm import tqdm


def get_chunks_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    text = ' '.join([data.get_text() for data in parsed_response.find_all('p')])
    text = sub(r'\[[0-9\]]*', '', text)
    text = sub(r'\n', '', text)
    text = sub(r'\t', '', text)
    # removing footnotes - they are particularly frequent in Wikipedia articles
    sentences = tokenize.sent_tokenize(text)
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

MULTIMEDIA_FORMAT = {'png', 'jpg', 'gif', 'mid', 'midi', 'ogg'}
def get_links_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    link_objects = parsed_response.find(id='bodyContent').find_all('a')
    bare_links = [link.get('href') for link in link_objects
                  if link.get('href') and link.get('href').split('.')[-1].lower() not in MULTIMEDIA_FORMAT]
    return ['https://ar.wikipedia.org' + link for link in bare_links]

def main():
    all_chunks = []
    visited_sites = set()
    reachable_sites = set()
    reachable_sites.add('https://ar.wikipedia.org/wiki/المجموعة_الشمسية')
    
    data_size = 7300
    progress = tqdm(total=data_size)
    while len(all_chunks) < data_size:
        try:
            http_response = requests.get(url=reachable_sites.pop())
        except requests.ConnectionError:
            continue
        
        if len(reachable_sites) < 1000:
            new_reachable_sites = get_links_from_response(http_response)
            for site in new_reachable_sites:
                if site not in visited_sites:
                    visited_sites.add(site)
                    reachable_sites.add(site)
        for chunk in get_chunks_from_response(http_response):
            all_chunks.append(chunk)
            progress.update(1)
    final_json = json.dumps([{'paragraph': paragraph.rstrip(), 'language': 'arabic'} for paragraph in all_chunks], indent=4, ensure_ascii=False)
    with open("arabic.json", "w", encoding='utf8') as file:
        file.write(final_json)


if __name__ == '__main__':
    main()

