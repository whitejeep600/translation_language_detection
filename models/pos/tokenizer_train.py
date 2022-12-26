import json
from datasets import load_dataset
from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    processors,
)

def get_dataset():
    with open("data/train_pos.json", "r", encoding='utf8') as file:
        all_paragraphs = json.load(file)
    dataset = ""
    for p in all_paragraphs:
        dataset += p['paragraph'] + "\n"
    
    with open("data/train_tokenizer.txt", "w", encoding='utf8') as file:
        file.write(dataset)
    
def train_tokenizer():
    tokenizer = Tokenizer(models.WordLevel())
    # tokenizer.truncation = 
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    trainer = trainers.WordLevelTrainer(vocab_size=300, special_tokens=special_tokens)
    tokenizer.train(["data/train_tokenizer.txt"], trainer=trainer)
    
    sep_token_id = tokenizer.token_to_id("</s>")
    cls_token_id = tokenizer.token_to_id("<s>")
    tokenizer.post_processor = processors.RobertaProcessing(
        ("</s>", sep_token_id), ("<s>", cls_token_id),
        add_prefix_space=False
    )


    test = "DET NOUN ADP DET NOUN ADP DET ADJ NOUN ADP DET PROPN NOUN VERB PUNCT"
    encoding = tokenizer.encode(test)
    print(encoding.tokens)
    print(encoding.ids)

    tokenizer.save("tokenizer/tokenizer.json")


if __name__ == '__main__':
    # get_dataset()
    train_tokenizer()