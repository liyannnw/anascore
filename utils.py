
import json
# import re

# import nltk
from nltk.tokenize import WordPunctTokenizer

# import contractions


tokenizer= WordPunctTokenizer()

def read_argv():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("--concept_dict",action="store",
                        dest="concept_dict",help="path to conceptual information")

    return parser.parse_args()

# def normalize_node(label):
#     article_pattern = r'^(a|an|the)\s+'
#     normalized_node = re.sub(article_pattern, '', node, flags=re.IGNORECASE)
#     normalized_node = normalized_node.lower()
#     return normalized_node

def load_conceptnet_dict(file):
    dict_=dict()

    for line in file:
        json_line = json.loads(line.strip())
        for text, info in json_line.items():
            if text not in dict_: dict_[text] = None
            # for relation_name,_ in info.items():
            #     relations.add(relation_name)
            dict_[text] = info
    
    return dict_


def deduplicate(data):
   return list(map(list, set(map(tuple, data))))

# def tokenize_sentence(sentence):
#     # words = nltk.word_tokenize(sentence)
#     words = tokenizer.tokenize(sentence)
#     return " ".join(words)
def tokenize(sentence,lowercase=True):
    # words = nltk.word_tokenize(sentence)
    # expanded_sentence = contractions.fix(sentence)
    words = tokenizer.tokenize(sentence)
    if lowercase:
        words = [w.lower() for w in words]
    return words


