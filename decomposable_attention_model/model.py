import warnings
import itertools

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import numpy as np


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


class MLP(chainer.Chain):
    def __init__(self, in_size, n_units, dropout, activation=F.tanh,
                 activation_on_final=False):
        n_units = [in_size, ] + n_units
        params = {}
        for i, n in enumerate(pairwise(n_units)):
            # Initial W according the paper
            params['l' + str(i)] = L.Linear(*n, initialW=I.GlorotNormal())
        super(MLP, self).__init__(**params)
        self.add_persistent('_dropout', dropout)
        self.add_persistent('_activation_on_final', activation_on_final)
        self._activation = activation

    def __call__(self, x, train=True):
        layers = list(self.namedlinks(skipself=True))
        for i in xrange(len(layers)):
            x = self['l' + str(i)](x)
            if (self._activation is not None and
                    (i != len(layers) or self.activation_on_final)):
                x = self._activation(x)
            if train and self._dropout > 0.:
                x = F.dropout(x, self._dropout, train=train)
        return x


class DecomposableAttentionModel(chainer.Chain):
    def __init__(self, w2v, n_class, f_units, g_units, f_dropout, g_dropout,
                 emb_proj_units=None):
        feat_size = w2v.shape[1]
        if emb_proj_units is None:
            emb_proj_units = feat_size
        super(DecomposableAttentionModel, self).__init__(
            emb=L.EmbedID(w2v.shape[0], feat_size, initialW=w2v,
                          ignore_label=-1),
            f=MLP(emb_proj_units, [f_units, f_units], f_dropout,
                  activation_on_final=False),
            g=MLP(emb_proj_units * 2, [g_units, g_units], g_dropout,
                  activation_on_final=False),
            h=MLP(g_units * 2, [200, n_class], 0.2,
                  activation_on_final=False)
        )
        if emb_proj_units is None:
            self.add_persistent('_train_embedding', False)
        else:
            self.add_persistent('_train_embedding', True)
            self.add_link('emb_proj', MLP(feat_size, [emb_proj_units, ], 0.),)

    @staticmethod
    def _length_aware_softmax(e, l0, l1, xp):
        # e: (B, T0, T1)
        bs, t0, t1 = e.shape
        l0 = l0.reshape((bs, 1, 1))
        l1 = l1.reshape((bs, 1, 1))
        mask0 = (xp.tile(xp.arange(t0).reshape(1, t0, 1), (bs, 1, 1)) < l0).astype(e.dtype)
        mask1 = (xp.tile(xp.arange(t1).reshape(1, t1, 1), (bs, 1, 1)) < l1).astype(e.dtype)
        mask = (xp.matmul(mask0, mask1.swapaxes(1, 2))).astype(np.bool)
        # mask: (B, T0, T1)
        mask = chainer.Variable(mask)
        padding = chainer.Variable(xp.zeros(e.shape, dtype=e.dtype))

        e_max = F.max(e, keepdims=True)
        e_masked = F.where(mask, e, padding)
        e_masked = e_masked - F.broadcast_to(e_max, e.shape)

        e_sum0 = F.reshape(F.logsumexp(e_masked, axis=1), (bs, 1, t1))
        e_sum1 = F.reshape(F.logsumexp(e_masked, axis=2), (bs, t0, 1))

        s1 = F.exp(e_masked - F.broadcast_to(e_sum0, e.shape))
        s2 = F.exp(e_masked - F.broadcast_to(e_sum1, e.shape))
        s1 = F.where(mask, s1, padding)
        s2 = F.where(mask, s2, padding)
        return s1, s2

    @staticmethod
    def _token_wise_linear(x, f, l, train, xp):
        s = list(x.shape)
        n_tokens = np.prod(s[:-1])
        z = F.reshape(f(F.reshape(x, (n_tokens, -1)), train), s[:-1] + [-1,])
        if l is not None:
            # mask: (B, T, 1)
            mask = xp.tile(xp.arange(s[1]).reshape(1, s[1], 1), (s[0], 1, 1)) < l.reshape(s[0], 1, 1)
            mask = F.broadcast_to(chainer.Variable(mask), z.shape)
            padding = chainer.Variable(xp.zeros(z.shape, dtype=z.dtype))
            z = F.where(mask, z, padding)
        return z

    def _compare(self, a, beta, l, train, xp):
        # Make one comparison (correspond to eq. 3)
        # [(B, Ti, M), (B, Ti, M)] -> (B, Ti, M + M))
        concated = F.concat((a, beta), axis=2)
        # (B, Ti, M + M) -> (B, Ti, M')
        v_i = self._token_wise_linear(concated, self.g, l, train, xp)
        # (B, Ti, M') -> (B, M')
        v = F.sum(v_i, axis=1)
        return v

    def __call__(self, x0, x1, l0, l1, train=True):
        """ Forward computation.

        Args:
            x0: Chainer variable in shape (B, T0) where B is the batch size,
                T is the number of tokens in each data. Each element should be
                given as the index of embedding.
            x1: Chainer variable in shape (B, T1)

        Returns:

        """
        t0 = x0.shape[1]
        t1 = x1.shape[1]
        # a: (B, T0, M)
        a = self.emb(x0)
        # b: (B, T1, M)
        b = self.emb(x1)

        if not self._train_embedding:
            a.unchain_backward()
            b.unchain_backward()
            a = self._token_wise_linear(a, self.emb_proj, l0, train, self.xp)
            b = self._token_wise_linear(b, self.emb_proj, l1, train, self.xp)
        # Apply perceptron layer to each feature vectors ... eq. 1
        # (B, Ti, M) -> (B * Ti, M) -> (B * Ti, F) -> (B, Ti, F)
        a_f = self._token_wise_linear(a, self.f, l0, train, self.xp)
        b_f = self._token_wise_linear(b, self.f, l1, train, self.xp)
        # for each batch, calculate a_f[b]
        # e: (B, T0, T1)
        e = F.batch_matmul(a_f, b_f, transb=True)

        # att_*: (B, T0, T1)
        att_b, att_a = self._length_aware_softmax(e, l0, l1, self.xp)
        # sum((B, T0, T1).(B, T0, T1, M)) -> beta: (B, T0, M) ... eq. 2
        b_tiled = F.tile(F.expand_dims(b, 1), (1, t0, 1, 1))
        att_b = F.expand_dims(att_b, 3)
        beta = F.sum(F.broadcast_to(att_b, b_tiled.shape) * b_tiled, axis=2)
        # sum((B, T0, T1).(N, T0, T1, M)) -> beta: (B, T1, M) ... eq. 2
        a_tiled = F.tile(F.expand_dims(a, 2), (1, 1, t1, 1))
        att_a = F.expand_dims(att_a, 3)
        alpha = F.sum(F.broadcast_to(att_a, a_tiled.shape) * a_tiled, axis=1)

        # Make comparison, [(B, Ti, M), (B, Ti, M)] -> (B, M')
        v1 = self._compare(a, beta, l0, train, self.xp)
        v2 = self._compare(b, alpha, l1, train, self.xp)

        # (B, M' + M') -> (B, n_class)  ... eq. 4 & 5
        v = F.concat((v1, v2), axis=1)
        y = self.h(v)

        return y
