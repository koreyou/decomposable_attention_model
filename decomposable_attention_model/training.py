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
    if train:
        loss.backward()
        optimizer.update()
        model.global_step += len(batch)
    return float(to_cpu(loss.data)), acc, pred


def forward_pred(model, dataset, device=None):
    loss = Accumulator()
    acc = Accumulator()
    pred = []
    iterator = chainer.iterators.SerialIterator(dataset, batch_size=4,
                                                repeat=False, shuffle=False)
    for batch in iterator:
        l, a, p = _run_batch(model, None, batch, device, False)
        loss.add(l, len(batch))
        acc.add(a, len(batch))
        pred.append(p)
    pred = np.concatenate(pred)
    return loss.eval(), acc.eval(), pred


def train(model, optimizer, train_itr, n_epoch, dev=None, device=None,
          report_intv=None, tmp_dir='tmp.model', short_report_intv=None,
          lr_decay=0.9, lr_decay_intv=1000):
    loss = Accumulator()
    acc = Accumulator()
    min_loss = float('inf')
    min_epoch = 0
    orig_alpha = optimizer.alpha
    report_short_tmpl = "[   ] T/loss={:0.4f} T/acc={:0.2f}"
    report_tmpl = "[{:>3d}] T/loss={:0.4f} T/acc={:0.2f} D/loss={:0.4f} D/acc={:0.2f}"
    for batch in train_itr:
        l, a, _ = _run_batch(model, optimizer, batch, device, True)
        loss.add(l, len(batch))
        acc.add(a, len(batch))

        # long report
        if (report_intv is not None and
                model.global_step % report_intv > (report_intv - train_itr.batch_size)):
            # this is not executed at first epoch
            loss_dev, acc_dev, _ = forward_pred(model, dev, device=device)
            logging.info(report_tmpl.format(
                train_itr.epoch - 1, loss.eval(), acc.eval(), loss_dev, acc_dev))
            if loss_dev < min_loss:
                min_loss = loss_dev
                min_epoch = train_itr.epoch - 1
                chainer.serializers.save_npz(tmp_dir, model)
        elif (short_report_intv is not None and
              model.global_step % short_report_intv < train_itr.batch_size):
            logging.info(report_short_tmpl.format(l, a))
        if train_itr.epoch == n_epoch:
            break

        optimizer.alpha = orig_alpha * lr_decay ** (model.global_step / float(lr_decay_intv))
    logging.info('loading early stopped-model at epoch {}'.format(min_epoch))
    chainer.serializers.load_npz(tmp_dir, model)


class Accumulator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.val = 0.
        self.size = 0

    def add(self, val, size):
        self.val += val * size
        self.size += size

    def eval(self):
        val = self.val / float(self.size)
        self.clear()
        return val
