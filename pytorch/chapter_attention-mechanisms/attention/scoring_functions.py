# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-11-21
# @author: cnzero

"""mask-based scoring functions and classes:
`AdditiveAttention`, `DotProductAttention`
"""

import math
import torch
from torch import nn


def sequence_mask(X,
                  valid_len=None,
                  value=0.0):
    """mask a sequence out of `valid_len` with `value`

    Args:
        X (tensor): tensor to be truncated, shape(a, maxlen)
        valid_len (tensor): valid length to truncate the input tensor `X`
                            shape(a, )
        value (float, optional): Defaults to 0.0 to assign at the masking place

    Returns:
        tensor: truncated/masked tensor with the same shape of inputs `X`
                shape(a, b)
    """
    maxlen = X.size(1)
    # shape(1, maxlen)
    idx = torch.arange(start=0, end=maxlen, step=1,
                       dtype=torch.float32,
                       device=X.device)[None, :]
    # shape(a, 1)
    valid_lens = valid_len[:, None]
    # shape(a, maxlen)
    mask_boolmat = idx < valid_lens
    X[~mask_boolmat] = value

    return X


def masked_softmax(X,
                   valid_lens=None,
                   value=-1e6):
    """mask a mat `X` out of `valid_lens` with `value`

    Args:
        X (tensor): tensor to be truncated, shape(B, no.Q, no.KV)
        valid_lens (tensor): valid length to truncate the input tensor `X`
                             optional to None, .dim() = 1, 2, or 3
        value (float, optional): Defaults to -1e6 to assign at the masking place

    Returns:
        tensor: truncated/masked tensor with the same shape of inputs `X`
    """
    # case 0: valid_lens is None, it is O.K.
    # case 1: valid_lens.dim() == 1, assert len(valid_lens) == X.shape[0]
    # case 2: valid_lens.dim() == 2, 
    #         assert valid_lens.shape[0] == X.shape[0]
    #     and assert valid_lens.shape[1] == X.shape[1]
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            assert len(valid_lens) == X.shape[0]
            # shape(shape[0]*shape[1], )
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:  # .dim() == 2
            assert valid_lens.shape[0] == X.shape[0]
            assert valid_lens.shape[1] == X.shape[1]
            # shape(shape[0]*shape[1], )
            valid_lens = valid_lens.reshape(-1)  # flatten
        # finally, valid_lens.shape -->> (shape[0]*shape[1], )

        # -> X.shape(shape[0]*shape[1], shape[2])
        X = X.reshape(-1, shape[-1])
        X = sequence_mask(X=X,
                          valid_len=valid_lens,
                          value=value)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self,
                 query_size,
                 key_size,
                 num_hiddens,
                 dropout=0.0,
                 **kwargs):
        """init setting of the attention model

        Args:
            query_size ([type]): [description]
            key_size ([type]): [description]
            num_hiddens ([type]): [description]
            dropout ([type]): [description]
        """
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(in_features=query_size,
                             out_features=num_hiddens,
                             bias=False)
        self.W_k = nn.Linear(in_features=key_size,
                             out_features=num_hiddens,
                             bias=False)
        self.W_v = nn.Linear(in_features=num_hiddens,
                             out_features=1,
                             bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                queries,
                keys,
                values,
                valid_lens):
        """[summary]

        Args:
            queries (tensor): tensor, shape (B, 1, query_size)
            keys (tensor): tensor, shape (B, 1, key_size)
            values (tensor): tensor, shape (B, 1, value_size)
            valid_lens (tensor): tensor

        Returns:
            tensor: shape (B, 1)
        """
        # shape(B, no.Q, query_size) -> (B, no.Q, num_hiddens)
        queries = self.W_q(queries)

        # shape(B, no.KV, key_size) -> (B, no.KV, num_hiddens)
        keys = self.W_k(keys)

        # queries: shape(B, no.Q,  num_hiddens) -> shape (B, no.Q,  1, num_hiddens)
        # keys:    shape(B, no.KV, num_hiddens) -> shape (B, 1, no.KV, num_hiddens)
        # [browdcast] features shape(B, no.Q, no.KV, num_hiddens)
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))

        # shape(B, no.Q, no.KV, num_hiddens) * shape(num_hiddens, 1)
        # squeeze(-1) -> shape(B, no.Q, no.KV)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(X=scores,
                                                valid_lens=valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self,
                 dropout=0.0,
                 **kwargs):
        """[summary]

        Args:
            dropout ([type]): [description]
        """
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self,
                queries,
                keys,
                values,
                valid_lens=None):
        """[summary]

        Args:
            queries ([type]): [description]
            keys ([type]): [description]
            values ([type]): [description]
            valid_lens ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(X=scores,
                                                valid_lens=valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == "__main__":
    print('Hello world.')
    print('Wow, I bought a second-hand desk. I really need it.')

    print('-' * 20, 'test `masked_softmax()`')
    size_inputs = (2, 3, 4)
    # valid_lens = torch.tensor([2]) # invalid
    valid_lens = torch.tensor([2, 3])  # valid
    # valid_lens = torch.tensor([2, 3, 4]) # invalid
    # valid_lens = torch.tensor([[1,2,3], [2,3,4]]) # valid
    print(masked_softmax(X=torch.rand(size=size_inputs),
                         valid_lens=valid_lens))

    print('=' * 20, 'test `AdditiveAttention()`')
    batch_size = 2  # `batch`, single, rather than plural `batches`

    num_queries = 3  # `_queries`, plural, rather than `query`,
    # like `in_features` `num_hiddens` in standard torch APIs
    query_size = 20  # like `batch_size`, single

    key_size = 3  # like `batch_size`, single
    value_size = 6  # like `batch_size`, single
    num_kvs = 9  # like `num_hiddens`, plural key_value pairs

    num_hiddens = 8

    # shape(B, no.Q, query_size)
    queries = torch.normal(mean=0,
                           std=1,
                           size=(batch_size, num_queries, query_size))
    # shape(B, no.KV, key_size)
    keys = torch.ones(size=(batch_size, num_kvs, key_size))
    values = torch.arange(num_kvs * value_size,
                          dtype=torch.float32)
    # shape(B, no.KV, value_size)
    values = values.reshape((1, num_kvs, value_size)).repeat(repeats=(batch_size, 1, 1))

    valid_lens = torch.tensor([2, 2])

    attention = AdditiveAttention(query_size=query_size,
                                  key_size=key_size,
                                  num_hiddens=num_hiddens,
                                  dropout=0.0)
    attention.eval()
    print(attention(queries, keys, values, valid_lens).shape)

    print('+' * 20, 'test `DotProductAttention()`')
    query_size = key_size = d = 2
    queries = torch.normal(mean=0,
                           std=1,
                           size=(batch_size, num_kvs, d))
    keys = torch.ones(size=(batch_size, num_kvs, key_size))
    values = torch.arange(num_kvs * value_size,
                          dtype=torch.float32)
    values = values.reshape((1, num_kvs, value_size)).repeat(repeats=(batch_size, 1, 1))

    print('queries shape: ', queries.shape)
    print('keys shape:    ', keys.shape)
    print('values shape:  ', values.shape)

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens).shape)
