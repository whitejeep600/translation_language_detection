import stanza
import torch

from constants import CONVOLUTION_LENGTH, POS_TO_INT, NUM_POS_TAGS, D, \
    MAX_SENTENCE_LENGTH


pipeline = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)


def process_tree(tree):
    return [{'pos': word.words[0].upos, 'parent': word.words[0].head - 1} for word in tree.sentences[0].tokens]


def parse_sentence(sentence):
    tree = pipeline(sentence)
    return process_tree(tree)


# word representation is obtained by concatenating CONVOLUTION_LENGTH vectors,
# of which the i-th corresponds to the i-th ancestor of the word in the dependency
# tree. Each i-th vector is either a vector of zeroes (if i > distance from the word
# to the root, or the i-th ancestor's POS tag was not recognized) or a one-hot
# vector representing the i-th ancestor's POS tag (as per the POS->int mapping in
# POS_TO_INT).
# this function takes a parse tree and a word index in the original sentence,
# and returns a list whose i-th element is either -1 (if the i-th vector to be
# concatenated should be all zeroes) or n if the n-th feature of the i-th vector
# should be 'hot'.
def pos_indices_for_word(dependency_information, word_index):
    indices = []
    for i in range(CONVOLUTION_LENGTH):
        if word_index == -1:
            indices.append(-1)
        else:
            indices.append(POS_TO_INT[dependency_information[word_index]['pos']])
            word_index = dependency_information[word_index]['parent']
    return indices


# converts the list returned by the previous function to the list of indices that should
# be 'hot' in the vector resulting from the concatenation.
def pos_indexes_to_hot_features(pos_indices):
    hot_features = []
    for i in range(len(pos_indices)):
        if pos_indices[i] != -1:
            hot_features.append(i * NUM_POS_TAGS + pos_indices[i])
    return hot_features


# the first dimension corresponds to the representation features of the given word,
# the second - to the word's position in the sentence.
# i.e. the matrix contains vector representations of each word, glued together horizontally.
def sentence_to_matrix(sentence):
    parsed = parse_sentence(sentence)
    hot_feature_list = [pos_indexes_to_hot_features(pos_indices_for_word(parsed, i)) for i in range(len(parsed))]
    matrix = torch.zeros(D, MAX_SENTENCE_LENGTH, dtype=torch.float)
    for i in range(min([len(hot_feature_list), MAX_SENTENCE_LENGTH])):
        for feature in hot_feature_list[i]:
            matrix[feature, i] = torch.LongTensor([1.0])
    return matrix
