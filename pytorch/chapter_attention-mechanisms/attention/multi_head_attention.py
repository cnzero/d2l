# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-11-27
# @author: cnzero

"""hold on, deeply think,
and add some document descriptions for the script.
"""

import math
import torch
from torch import nn

from scoring_functions import DotProductAttention


def transpose_qkv(X, num_heads):
    """

    Args:
        X (tensor): shape(B, no.Q or no.KV, num_hiddens)
        num_heads (int): how many heads in the multi-head attention

    Returns:
        tensor: transposed inputs tensor for efficiently parallel computations
                shape(B*num_heads, no.Q or no.KV, num_hiddens/num_heads)

    """
    assert X.shape[-1] % num_heads == 0
    # shape(B, no.Q or no.KV, num_hiddens), to
    # shape(B, no.Q or no.KV, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # -> shape(B, num_heads, no.Q or no.KV, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # -> shape(B*num_heads, no.Q or no.KV, num_hiddens/num_heads)
    X = X.reshape(-1, X.shape[2], X.shape[3])

    return X


def transpose_output(X,
                     num_heads):
    """reverse the operation of `transpose_qkv`

    Args:
        X (tensor): shape(B*num_heads, no.Q or no.KV, num_hiddens/num_heads)
        num_heads (int): how many heads in the multi-head attention

    Returns:
        tensor: shape(B, no.Q or no.KV, num_hiddens)

    """
    # shape(B*num_heads, no.Q or no.KV, num_hiddens/num_heads), to
    # shape(B, num_heads, no.Q or no.KV, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

    # -> shape(B, no.Q or no.KV, num_heads, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # -> shape(B, no.Q or no.KV, num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], -1)

    return X


class MultiHeadAttention(nn.Module):
    """multi-head attention"""

    def __init__(self,
                 query_size,
                 key_size,
                 value_size,
                 mha_num_hiddens,
                 mha_num_heads,
                 dropout=0.0,
                 bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        assert mha_num_hiddens % mha_num_heads == 0
        self.mha_num_heads = mha_num_heads

        self.attention = DotProductAttention(dropout=dropout)
        self.W_q = nn.Linear(in_features=query_size,
                             out_features=mha_num_hiddens,
                             bias=bias)
        self.W_k = nn.Linear(in_features=key_size,
                             out_features=mha_num_hiddens,
                             bias=bias)
        self.W_v = nn.Linear(in_features=value_size,
                             out_features=mha_num_hiddens,
                             bias=bias)
        self.W_0 = nn.Linear(in_features=mha_num_hiddens,
                             out_features=mha_num_hiddens,
                             bias=bias)

    def forward(self,
                queries,
                keys,
                values,
                valid_lens):
        queries = transpose_qkv(X=queries,
                                num_heads=self.mha_num_heads)

        keys = transpose_qkv(X=keys,
                             num_heads=self.mha_num_heads)

        values = transpose_qkv(X=values,
                               num_heads=self.mha_num_heads)

        if valid_lens is not None:
            # repeat `mha_num_heads` rows
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.mha_num_heads,
                                                 dim=0)
        output = self.attention(queries=queries,
                                keys=keys,
                                values=values,
                                valid_lens=valid_lens)
        output_concat = transpose_output(X=output,
                                         num_heads=self.mha_num_heads)
        return self.W_0(output_concat)


if __name__ == '__main__':
    print('Hello world in multi_head_attention.py')
    print('-'*20, 'MultiHeadAttention().')

    num_hiddens = query_size = key_size = value_size = 100
    num_heads = 5

    mha = MultiHeadAttention(query_size=query_size,
                             key_size=key_size,
                             value_size=value_size,
                             mha_num_hiddens=num_hiddens,
                             mha_num_heads=num_heads)
    mha.eval()

    batch_size = 2
    num_queries = 4
    num_kvs = 6
    valid_lens = torch.tensor([3, 2])
    queries = torch.ones(size=(batch_size, num_queries, query_size))
    keys = torch.ones(size=(batch_size, num_kvs, key_size))
    values = torch.ones(size=(batch_size, num_kvs, value_size))

    output = mha(queries=queries,
                 values=values,
                 keys=keys,
                 valid_lens=valid_lens)
    print(output.shape)
