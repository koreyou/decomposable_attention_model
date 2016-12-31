import argparse
import logging

import chainer

from decomposable_attention_model import dataset, word2vec, training
from decomposable_attention_model.model import DecomposableAttentionModel


logging.basicConfig(level=logging.INFO)

def run(args):
    vocab_list = dataset.aggregate_vocabs([args.train, args.test])
    logging.info("Loading word2vec")
    w2v, vocab, ind_unks = word2vec.load_word2vec(args.word2vec, 20,
                                                  vocab_list=vocab_list)

    train = dataset.create_dataset(args.train, vocab, ind_unks)
    logging.info("loaded {} lines for training".format(len(train)))

    dev = dataset.create_dataset(args.dev, vocab, ind_unks, 500)
    test = dataset.create_dataset(args.test, vocab, ind_unks, -1)
    dam = DecomposableAttentionModel(
        w2v, 3, f_units=200, g_units=200, f_dropout=0.2,
        g_dropout=0.2, emb_proj_units=None)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(dam)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.))
    if args.gpu is not None:
        dam.to_gpu(device=args.gpu)

    train_itr = chainer.iterators.SerialIterator(train, batch_size=4)
    training.train(dam, optimizer, train_itr, 10, dev=dev,
                   device=args.gpu)
    loss, acc, _ = training.forward_pred(dam, test, device=args.gpu)
    logging.info("Test => loss={:0.4f} acc={:0.2f}".format(loss, acc))
    if args.gpu is not None: dam.to_cpu()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, type=str,
                   help='SNLI train json file path')
    p.add_argument('--dev', required=True, type=str,
                   help='SNLI dev json file path')
    p.add_argument('--test', required=True, type=str,
                   help='SNLI test json file path')
    p.add_argument('--word2vec', required=True, type=str,
                   help='Word2vec pretrained file path')

    # optional
    p.add_argument('-g', '--gpu', type=int, default=None, help="GPU number")
    args = p.parse_args()

    run(args)