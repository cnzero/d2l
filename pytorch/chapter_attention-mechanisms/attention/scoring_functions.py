import math
import torch
from torch import nn


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen, 
                        dtype=torch.float32, 
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X=X.reshape(-1, shape[-1]), 
                          valid_len=valid_lens, 
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=1)

class AdditiveAttention(nn.Module):
    def __init__(self, 
                 query_size, 
                 key_size,
                 num_hiddens, 
                 dropout, 
                 **kwargs):
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
                keyes, 
                values, 
                valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keyes)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(X=scores, 
                                                valid_lens=valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    def __init__(self, 
                 dropout, 
                 **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                queries, 
                keys, 
                values, 
                valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(X=scores, 
                                                valid_lens=valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == "__main__":
    print('Hello world.')
    print('Wow, I bought a second-hand desk. I really need it.')

    print('-'*20, 'test `masked_softmax()`')
    print(masked_softmax(X=torch.rand(size=(2, 2, 4)), 
                         valid_lens=torch.tensor([2, 3])))
