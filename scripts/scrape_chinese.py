import json
import requests
from bs4 import BeautifulSoup
from re import sub, fullmatch
from nltk import tokenize
from tqdm import tqdm
import random


def get_chunks_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    content = parsed_response.find(id='bodyContent')
    if content == None:
        return []
    text = ''.join([data.get_text() for data in content.find_all('p')])

    # removing footnotes - they are particularly frequent in Wikipedia articles
    text = sub(r'\[[a-z]*.*[0-9]*\]', '', text)
    text = sub(r'\n', '', text)
    text = sub(r'\t', '', text)
    text = sub(r'\[來源請求\]', '', text)

    # filter the text
    if '本頁面是一個維護分類' in text or '自由百科全書' in text:
        return []

    if '维基百科' in text or '維基百科' in text or '優良條目評選' in text:
        return []

    if '原始文件' in text:
        return []

    sentences = [sentence + "。" for sentence in text.split("。")]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # filter the sentence
        if '维基百科' in sentence: # if 'wikipedia' in sentence
            continue

        if len(current_chunk) + len(sentence) < 256:
            current_chunk += sentence
        else:
            current_chunk += sentence
            if (len(chunks) < 3):
                chunks.append(current_chunk)
                current_chunk = ''
            else:
                break
    #print(chunks)
    return chunks

MULTIMEDIA_FORMAT = {'png', 'jpg', 'gif', 'mid', 'midi', 'ogg'}
def get_links_from_response(http_response):
    parsed_response = BeautifulSoup(http_response.content, 'html.parser')
    content = parsed_response.find(id='bodyContent')
    if content == None:
        return []
    link_objects = content.find_all('a')
    bare_links = [link.get('href') for link in link_objects
                  if link.get('href') and link.get('href').split('.')[-1].lower() not in MULTIMEDIA_FORMAT]

    bare_links = random.sample(bare_links, len(bare_links) // 5) # limit the breadth of the BFS by random sample
    return ['https://zh.wikipedia.org' + link for link in bare_links]

def main():
    random.seed()
    all_chunks = []
    visited_sites = set()
    reachable_sites = set()
    reachable_sites.add('https://zh.wikipedia.org/wiki/%E5%93%B2%E5%AD%A6')

    data_size = 150
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
            #print(chunk)
            if chunk not in all_chunks:
                all_chunks.append(chunk)
                progress.update(1)

    random.shuffle(all_chunks) # shuffle the list
    final_json = json.dumps([{'paragraph': paragraph.rstrip(), 'language': 'chinese'} for paragraph in all_chunks], indent=4, ensure_ascii=False)
    with open("chinese.json", "w", encoding='utf8') as file:
        file.write(final_json)


if __name__ == '__main__':
    main()