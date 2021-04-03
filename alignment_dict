import argparse
import random
from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np

random.seed = 0

parser = argparse.ArgumentParser(description="Make a dictionary for embedding alignments.")
parser.add_argument('--dict_file', help="Path to the dictionary made from dictionary_file.py")
parser.add_argument('--embeddings_lang1', help="Path to the embedding file of the source language in word2vec format")
parser.add_argument('--embeddings_lang2', help="Path to the embedding file of the target language in word2vec format")
parser.add_argument('--sense_embeddings', action='store_true',
                    help="Indicate if the vectors are sense-disambiguated by sensegram")
parser.add_argument('--vocab',
                    help="Path to a vocab file in the source language, e.g. from GloVe, with lines of <word> <count>")
args = parser.parse_args()


def read_dict_file(dict_file):
    print("Reading dict file ", dict_file)
    f = open(dict_file)
    translations = {}
    inv_translations = {}

    for line in f:
        x, y = tuple(line.split())

        try:
            translations[x].append(y)
        except KeyError:
            translations[x] = [y]

        try:
            inv_translations[y].append(x)
        except KeyError:
            inv_translations[y] = [x]

    return translations, inv_translations


def get_words(embedding_file):
    print("Reading embeddings ", embedding_file)
    model = KeyedVectors.load_word2vec_format(embedding_file)
    words = set(model.vocab)
    return words


def read_vocab(vocab_file):
    print("Reading vocab file ", vocab_file)
    f = open(vocab_file)
    vocab = defaultdict(lambda: 0)

    for line in f:
        try:
            x, y = tuple(line.split())
            vocab[x] = int(y)
        except ValueError:
            pass

    return vocab


if __name__ == '__main__':

    if args.sense_embeddings:

        # collect translations that satisfy these criterions:
        # - source word frequency is above the median
        # - source word frequency is below (0.9 * max frequency)
        # - source word has only 1 sense in the model
        # - translation word(s) have only 1 sense in the model

        words1 = get_words(args.embeddings_lang1)
        words2 = get_words(args.embeddings_lang2)
        vocab = read_vocab(args.vocab)
        trans, inv = read_dict_file(args.dict_file)

        min_count = np.mean(list(vocab.values()))
        max_count = max(list(vocab.values())) * 0.9

        print("Collecting translations...")
        good_train_keys = [k for k in trans.keys() if min_count < vocab[k] < max_count and k + "#2" not in words1]
        good_train_trans = {}
        for k in good_train_keys:
            good_train_trans[k] = [t for t in trans[k] if t + "#2" not in words2]
        good_train_trans = {k: t for k, t in good_train_trans.items() if t != []}

        # testing can be done with multi-sense words
        # but the frequency limit is the same

        good_test_keys = [k for k in trans.keys() if min_count < vocab[k] < max_count]
        good_test_trans = {}
        for k in good_test_keys:
            good_test_trans[k] = [t for t in trans[k]]

        # randomly select train and test sets
        print("Random choices...")

        train_trans_set = random.choices(list(good_train_trans.keys()), k=5000)
        test_trans_set = random.choices(list(set(good_test_trans.keys()) - set(train_trans_set)), k=2000)

        # write dictionaries to files
        print("Writing files...")

        f = open(args.dict_file + ".train", 'w')
        for k in train_trans_set:
            for t in good_train_trans[k]:
                f.write(f"{k} {t}\n")
        f.close()

        f = open(args.dict_file + ".test", 'w')
        for k in test_trans_set:
            for t in good_test_trans[k]:
                f.write(f"{k} {t}\n")
        f.close()

