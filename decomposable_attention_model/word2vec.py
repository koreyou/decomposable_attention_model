import codecs

import numpy as np


def _get_infreq_word_index(vocab, thresh):
    freq = np.asarray([v.count for v in vocab])
    ind = np.asarray([v.index for v in vocab])
    # argsort returns index in accending order so use first n
    n = int(len(vocab) * thresh)
    return ind[np.argsort(freq)[:n]]


def _parse_line(line):
    data = line.strip().split(u' ')
    token = data[0]
    v = map(float, data[1:])
    return token, v


def _load(path, vocab_list=None):
    vocab = {}
    with codecs.open(path, mode='r', encoding='utf-8') as fin:
        arr = None
        i = 0
        for line in fin:
            token, v = _parse_line(line)
            if vocab_list is not None and token not in vocab_list:
                continue
            if arr is None:
                arr = np.array(v, np.float32).reshape(1, -1)
            else:
                arr = np.append(arr, [v], axis=0)
            vocab[token] = i
            i += 1
    return arr, vocab


def load_word2vec(path, n_unks, vocab_list=None):
    """
    Load word2vec vectors from a pretrained model.
    Unknown vocab vector is assigned to v[vocab[u'<unk>'], :].
    Unknown voab is calculated by averaging all infrequent words.

    Args:
        path (str): path to a pretrained model.

    Returns:
        numpy.ndarray: Distributed representation
        dict: unicode vocab to index mapping.
        list: mapping to unks

    """
    mat, vocab = _load(path, vocab_list)
    v_unk = np.random.normal(scale=1., size=(n_unks, mat.shape[1]))
    v_unk = v_unk / np.linalg.norm(v_unk, axis=1, keepdims=True)
    ind_unk = range(len(mat), len(mat) + n_unks)
    mat = np.vstack((mat, v_unk))
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return mat, vocab, ind_unk
