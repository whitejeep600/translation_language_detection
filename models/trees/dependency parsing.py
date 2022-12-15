import stanza


def parse_sentence(sentence):
    pipeline = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')
    tree = pipeline(sentence)
    return tree


if __name__ == '__main__':
    print(parse_sentence("your father jumps over your lazy mother"))
