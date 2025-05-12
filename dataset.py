import os, gzip, torch
import wget
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import Counter, OrderedDict
import random

def load_imdb(final=False, val_size=5000):
    """
    Loads the imdb dataset. Downloads it if necessary.

    :param final: If true, return the canonical test/train split (25k training, 25k test)
                 If false, split off a validation set of val_size instances from the training data
    :param val_size: Number of instances in the validation split.
    :return: (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes
    """

    url = 'https://github.com/pbloem/former/raw/master/data/imdb.tgz'
    name = 'imdb.tgz'

    if not os.path.exists(name):
        print('Downloading IMDB dataset.')
        wget.download(url)

    if not os.path.exists('imdb') or not os.path.isdir('imdb'):
        print('Extracting IMDB dataset.')
        os.system(f'tar xzf {name}')

    print('Loading IMDB dataset.')
    xtrain, ytrain = load_split('imdb/train')
    xtest, ytest = load_split('imdb/test')

    print('Computing vocabulary.')
    i2w, w2i = build_vocab(xtrain)

    print('Converting to indices.')
    xtrain = [sent2indices(s, w2i) for s in xtrain]
    xtest  = [sent2indices(s, w2i) for s in xtest]

    print('Sorting by length.')
    xtrain, ytrain = sort_by_len(xtrain, ytrain)
    xtest, ytest = sort_by_len(xtest, ytest)

    if not final: # return a validation split
        val_size = val_size
        xs = xtrain[0:val_size]
        ys = ytrain[0:val_size]

        xt = xtrain[val_size:]
        yt = ytrain[val_size:]

        return (xt, yt), (xs, ys), (i2w, w2i), 2

    return (xtrain, ytrain), (xtest, ytest), (i2w, w2i), 2

def load_split(dir):
    """
    Load a split of the IMDB dataset: a list of strings and a list of labels.
    :param dir:
    :return:
    """
    pos, neg = [], []

    for split in ['pos', 'neg']:
        for file in os.listdir(os.path.join(dir, split)):
            with open(os.path.join(dir, split, file)) as f:
                (neg if split == 'neg' else pos).append(f.read())

    data = pos + neg
    labels = [0] * len(pos) + [1] * len(neg)

    return data, labels

def build_vocab(data):
    """
    Builds a vocabulary for the IMDB dataset.
    :param data:
    :return:
    """
    tokens = []
    for s in data:
        tokens.extend(tokenize(s))

    counts = Counter(tokens)
    vocab = sorted([t for t in counts if counts[t] > 1])

    i2w = ['.pad', '.start', '.end', '.unk'] + vocab
    w2i = {t:i for i, t in enumerate(i2w)}

    return i2w, w2i

def tokenize(sentence):
    """
    Tokenize a string by splitting on non-alphanumeric characters and converting to lowercase.
    :param sentence:
    :return:
    """

    return [t.lower() for t in sentence.split() if len(t) > 0]

def clean(sentence):
    """
    Remove non-alphanumeric characters
    :param sentence:
    :return:
    """
    import re

    return re.sub(r'[^\w\s]', '', sentence).lower().strip()

def sent2indices(sentence, w2i):
    """
    Convert a sentence to a list of indices.
    :param sentence:
    :param w2i:
    :return:
    """

    l = [w2i[w] if w in w2i else w2i['.unk'] for w in tokenize(sentence)]

    return l

def sort_by_len(data, labels):
    """
    Sort a dataset by sentence length
    :param data:
    :param labels:
    :return:
    """
    data, labels = zip(*sorted(zip(data, labels), key=lambda p: len(p[0])))

    return list(data), list(labels)
