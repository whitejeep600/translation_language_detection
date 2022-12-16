import random

import stanza

from translation_language_detection.models.trees.train import read_training_data


def length_to_root(tree, index):
    if tree.sentences[0].tokens[index].words[0].head == 0:
        return 0
    else:
        return length_to_root(tree, tree.sentences[0].tokens[index].words[0].head - 1) + 1


def tree_height(tree):
    return max([length_to_root(tree, index) for index in range(len(tree.sentences[0].tokens))])


def parse_sentence(sentence, pipeline):
    tree = pipeline(sentence)
    return tree


if __name__ == '__main__':
    pipeline = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')
    # all_sentences = read_training_data()
    # random.shuffle(all_sentences)
    # sample = [sentence['text'] for sentence in all_sentences[:128]]
    # lengths = [tree_height(tree) for tree in [parse_sentence(sentence, pipeline) for sentence in sample]]
    # print(sorted(lengths))
    # print(sum(lengths)/len(lengths))
    # about 10% of height greater than 8
    print(parse_sentence("your father jumps over your very lazy mother", pipeline))
