import logging

import chainer
import chainer.functions as F
from chainer.cuda import to_cpu
import numpy as np


def _accuracy(pred, t):
    y = to_cpu(pred)
    t = to_cpu(t)
    return float(np.average(y == t))


def clip_data(x, l):
    return x[:, :max(l)]


def _run_batch(model, optimizer, batch, device, train):
    assert train == (optimizer is not None)
    model.cleargrads()
    x0, x1, l0, l1, t = chainer.dataset.concat_examples(batch, device=device)
    x0 = clip_data(x0, l0)
    x1 = clip_data(x1, l1)
    y = model(chainer.Variable(x0), chainer.Variable(x1), l0, l1, train)
    loss = F.softmax_cross_entropy(y, chainer.Variable(t))
    pred = to_cpu(F.argmax(F.softmax(y), axis=1).data)
    acc = _accuracy(pred, t)
    if optimizer is not None:
        loss.backward()
        optimizer.update()
    return float(to_cpu(loss.data)), acc, pred


def forward_pred(model, dataset, device=None):
    loss = 0.
    acc = 0.
    pred = []
    iterator = chainer.iterators.SerialIterator(dataset, batch_size=4,
                                                repeat=False, shuffle=False)
    for batch in iterator:
        l, a, p = _run_batch(model, None, batch, device, False)
        loss += l * len(batch)
        acc += a * len(batch)
        pred.append(p)
    pred = np.concatenate(pred)
    return loss / float(len(dataset)), acc / float(len(dataset)), pred


def train(model, optimizer, train_itr, n_epoch, dev=None, device=None,
          tmp_dir='tmp.model'):
    loss = 0.
    acc = 0.
    min_loss = float('inf')
    min_epoch = 0
    report_tmpl = "[{:>3d}] T/loss={:0.4f} T/acc={:0.2f} D/loss={:0.4f} D/acc={:0.2f}"
    for batch in train_itr:
        if train_itr.is_new_epoch:
            # this is not executed at first epoch
            loss_dev, acc_dev, _ = forward_pred(model, dev, device=device)
            loss = loss / len(train_itr.dataset)
            acc = acc / len(train_itr.dataset)
            logging.info(report_tmpl.format(
                train_itr.epoch - 1, loss, acc, loss_dev, acc_dev))
            if loss_dev < min_loss:
                min_loss = loss_dev
                min_epoch = train_itr.epoch - 1
                chainer.serializers.save_npz(tmp_dir, model)

            loss = 0.
            acc = 0.
        if train_itr.epoch == n_epoch:
            break
        l, a, _ = _run_batch(model, optimizer, batch, device, True)
        loss += l * len(batch)
        acc += a * len(batch)
    logging.info('loading early stopped-model at epoch {}'.format(min_epoch))
    chainer.serializers.load_npz(tmp_dir, model)
