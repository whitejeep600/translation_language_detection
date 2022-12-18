LANGUAGES = ['arabic', 'indonesian']

label_to_int = {
    'arabic': 0,
    'chinese': 1,
    'indonesian': 2,
    'japanese': 3
}

TEST_TRANSLATORS = ['helsinki', 'mbart']

NUM_LABELS = len(label_to_int)

CONVOLUTION_LENGTH = 8

POS_TO_INT = {
    'ADJ': 0,
    'ADP': 1,
    'ADV': 2,
    'AUX': 3,
    'CCONJ': 4,
    'DET': 5,
    'INTJ': 6,
    'NOUN': 7,
    'NUM': 8,
    'PART': 9,
    'PRON': 10,
    'PROPN': 11,
    'PUNCT': 12,
    'SCONJ': 13,
    'SYM': 14,
    'VERB': 15,
    'X': -1
}

NUM_POS_TAGS = len(POS_TO_INT) - 1  # not counting the 'X' ('other', 'unrecognized') tag

# word representation dimension
D = CONVOLUTION_LENGTH * NUM_POS_TAGS

# most sentences have at most 64 words, so we can truncate/pad to this length
MAX_SENTENCE_LENGTH = 64

NUM_EPOCH = 1

LEARNING_RATE = 1e-4

SAVE_DIR = 'checkpoint/model.pt'
