import json
import codecs
import random

import numpy as np
from chainer.datasets import TupleDataset


TEXT_TO_CLASS = {u'entailment':0, u'contradiction':1, u'neutral':2, u"-": 3}
CLASS_TO_TEXT = {0:u'entailment', 1:u'contradiction', 2:u'neutral'}


def _embed(line, vocab, ind_unks):
    line = line.split()
    tokens = []
    for t in line:
        if t in vocab:
            tokens.append(vocab[t])
        else:
            tokens.append(random.choice(ind_unks))
    tokens.append(vocab[u'</s>'])
    return tokens


def _parse_single(dic, vocab, ind_unks):
    label = TEXT_TO_CLASS[dic[u'gold_label']]
    if label == 3:
        # They give u"-" when the annotator consistency is low
        return None, None, None
    text = filter(lambda x: x not in u'()', dic[u'sentence1_binary_parse'])
    text = _embed(text, vocab, ind_unks)
    hypothesis = filter(lambda x: x not in u'()', dic[u'sentence2_binary_parse'])
    hypothesis = _embed(hypothesis, vocab, ind_unks)
    return text, hypothesis, label


def _pad_create(arr, dtype):
    s = max(map(len, arr))
    ret = np.zeros((len(arr), s), dtype=dtype)
    length = np.empty((len(arr), ), dtype=np.int32)
    for i, a in enumerate(arr):
        ret[i, :len(a)] = a
        length[i] = len(a)
    return ret, length


def create_dataset(path, vocab, ind_unks, size=-1):
    texts = []
    hypotheses = []
    labels = []
    with codecs.open(path, mode='r', encoding='utf-8') as fin:
        # Data has one json per line
        for i, line in enumerate(fin):
            d = json.loads(line)
            t, h, l = _parse_single(d, vocab, ind_unks)
            if t is None:
                continue
            texts.append(t)
            hypotheses.append(h)
            labels.append(l)
    texts, texts_len = _pad_create(texts, np.int32)
    hypotheses, hypotheses_len = _pad_create(hypotheses, np.int32)
    labels = np.array(labels, dtype=np.int32)
    if size > 0:
        # Sample data AFTER all data has been loaded. This is because
        # There might be bias in data ordering.
        ind = np.random.permutation(len(labels))[:size]
        return TupleDataset(texts[ind], hypotheses[ind], texts_len[ind],
                            hypotheses_len[ind], labels[ind])
    else:
        return TupleDataset(texts, hypotheses, texts_len, hypotheses_len, labels)


def aggregate_vocabs(paths):
    vocab = set([u'</s>'])
    for path in paths:
        if path is None:
            continue
        with codecs.open(path, mode='r', encoding='utf-8') as fin:
            # Data has one json per line
            for line in fin:
                d = json.loads(line)
                for tag in [u'sentence1_binary_parse', u'sentence2_binary_parse']:
                    text = filter(lambda x: x not in u'()', d[tag])
                    vocab.update(set(text.split()))
    return vocab
