import json
from pathlib import Path
import textstat
import nltk
from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import words
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

tags = None
alphabet = 'abcdefghijklmnopqrstuvwxyz'
def check_word(word):
    for c in alphabet:
        if c in word:
            return True
    return False
def LoadData(function):
    data = []
    ja_data = json.loads(Path(function + '_japanese.json').read_text())
    ch_data = json.loads(Path(function + '_chinese.json').read_text())
    ar_data = json.loads(Path(function + '_arabic.json').read_text())
    in_data = json.loads(Path(function + '_indonesian.json').read_text())
    data.extend(ja_data)
    data.extend(ch_data)
    data.extend(ar_data)
    data.extend(in_data)
    return data

def word_count(text):
    return textstat.lexicon_count(text, removepunct=True)
 
def sentence_count(text):
    return textstat.sentence_count(text)
 
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length
 


def automated_readability_index(paragraph):
    return textstat.automated_readability_index(paragraph)

def coleman_liau_index(paragraph):
    return textstat.coleman_liau_index(paragraph)

def type_token(paragraph):
    res = 0
    words = nltk.word_tokenize(paragraph)
    for word in words:
        if check_word(word):
            res += 1
    return res / len(words)

def pos_tag(paragraph):
    global tags
    l = nltk.word_tokenize(paragraph)
    text = nltk.Text(l)
    tags = nltk.pos_tag(text)

def num_ratio(paragraph):
    global tags
    counter = defaultdict(int)
    for word, tag in tags:
        counter[tag] += 1
    return counter['CD'] / len(tags)

def verb_ratio(paragraph):
    global tags
    res = 0
    for word, tag in tags:
        if 'VB' in tag:
            res += 1
    return res / len(tags)
    
def prep_ratio(paragraph):
    global tags
    counter = defaultdict(int)
    for word, tag in tags:
        counter[tag] += 1
    return counter['IN'] / len(tags)

def conj_ratio(paragraph):
    global tags
    counter = defaultdict(int)
    for word, tag in tags:
        counter[tag] += 1
    return counter['CC'] / len(tags) 

def noun_ratio(paragraph):
    global tags
    res = 0
    for word, tag in tags:
        if 'NN' in tag:
            res += 1
    return res / len(tags)

def count_close_words(paragraph):
    global tags
    counter = defaultdict(int)
    for word, tag in tags:
        counter[tag] += 1
    return counter['DT'] + counter['CD'] + counter['CC'] + counter['PDT'] + counter['POS'] + counter['TO'] + counter['WDT'] + counter['WDT'] + counter['WP'] + counter['WP$'] + counter['WRB']
def open_words_ratio(paragraph):
    global tags
    return (len(tags) - count_close_words(paragraph)) / len(tags)
def close_words_ratio(paragraph):
    global tags
    return count_close_words(paragraph) / len(tags)

def grammlex(paragraph):
    if close_words_ratio(paragraph) == 0:
      return 0
    return open_words_ratio(paragraph) / close_words_ratio(paragraph)

def dmark_ratio(paragraph):
    global tags
    counter = defaultdict(int)
    for word, tag in tags:
        counter[tag] += 1
    return counter['UH'] / len(tags) 

def pnoun_ratio(paragraph):
    global tags
    counter = defaultdict(int)
    for word, tag in tags:
        counter[tag] += 1
    return (counter['PRP'] + counter['PRP$']) / len(tags) 

def lemma_check(word):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
        if lemma == word:
            return True
    return False
def lemma_ratio(paragraph):
    res = 0
    words = nltk.word_tokenize(paragraph)
    for word in words:
        if lemma_check(word) and check_word(word):
            res += 1
    return res / len(words)

def Label(data):
  pos_tag(data)
  func = [avg_sentence_length, automated_readability_index, coleman_liau_index, type_token, num_ratio, verb_ratio, prep_ratio, conj_ratio, open_words_ratio, dmark_ratio, grammlex, pnoun_ratio, lemma_ratio, close_words_ratio]
  vec = []
  for f in func:
    vec.append(f(data))
  return vec
def LabelData(data):
    ans = {'japanese':0, 'indonesian':1, 'chinese':2, 'arabic':3}
    x, y = [], []
    for d in data:
      x.append(Label(d['paragraph']))
      y.append(ans[d['language']])
    return x, y 

if __name__ == '__main__':
    train_data = LoadData('train')
    test_data = LoadData('test')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

    train_x, train_y = LabelData(train_data)
    test_x, test_y = LabelData(test_data)

    linear = svm.SVC(kernel='linear', C=5).fit(train_x, train_y)
    linear_pred = linear.predict(test_x)

    linear_accuracy = accuracy_score(test_y, linear_pred)
    linear_f1 = f1_score(test_y, linear_pred, average='weighted')
    print('Accuracy (Linear Kernel): ', "%.2f" % (linear_accuracy*100))
    print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100))

    #linear
    remap = {0:'japanese',1:'indonesian',2:'chinese',3:'arabic'}
    else_counter = defaultdict(int)
    else_counter2 = defaultdict(int)
    mbart_counter = defaultdict(int)
    mbart_counter2 = defaultdict(int)
    counter = defaultdict(int)
    counter2 = defaultdict(int)
    for idx, d in enumerate(test_data):
        if 'mbart' in d['translator'].lower():
            mbart_counter[remap[test_y[idx]]] += 1
            if test_y[idx] == linear_pred[idx]:
                mbart_counter2[remap[test_y[idx]]] += 1
        else:
            else_counter[remap[test_y[idx]]] += 1
            if test_y[idx] == linear_pred[idx]:
                else_counter2[remap[test_y[idx]]] += 1
        counter[remap[test_y[idx]]] += 1
        if test_y[idx] == linear_pred[idx]:
            counter2[remap[test_y[idx]]] += 1
            
    print('-----mbart-----')    
    for i in remap:
        print(remap[i], mbart_counter2[remap[i]]/mbart_counter[remap[i]])

    print('-----else-----')    
    for i in remap:
        print(remap[i], else_counter2[remap[i]]/else_counter[remap[i]])

    print('-----overall-----')
    for i in remap:
        print(remap[i], counter2[remap[i]]/counter[remap[i]])