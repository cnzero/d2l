import math
import torch
from torch import nn

from scoring_functions import masked_softmax

class AdditiveAttention(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, 
                 query_size, 
                 key_size, 
                 num_hiddens, 
                 dropout):
        """[summary]

        Args:
            query_size ([type]): [description]
            key_size ([type]): [description]
            num_hiddens ([type]): [description]
            dropout ([type]): [description]
        """
        super(AdditiveAttention, self).__init__()

        self.W_q = nn.Linear(in_features=query_size, 
                             out_features=num_hiddens, 
                             bias=False)
        self.W_k = nn.Linear(in_features=key_size, 
                             out_features=num_hiddens, 
                             bias=False)
        self.w_v = nn.Linear(in_features=num_hiddens, 
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
            queries ([type]): [description]
            keys ([type]): [description]
            values ([type]): [description]
            valid_lens ([type]): [description]
        """
        # shape(B, query_size, 1, num_hiddens)
        queries = self.W_q(queries)
        # shape(B, 1, no. k-v pairs, num_hiddens)
        keys = self.W_k(keys)

        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(X=scores, 
                                                valid_lens=valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, 
                 dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

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
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

if __name__ == '__main__':
    queries = torch.normal(mean=0, 
                           std=1, 
                           size=(2, 1, 20))
    keys = torch.ones(size=(2, 10, 2))
    values = torch.arange(40, dtype=torch.float32)
    values = values.reshape(shape=(1, 10, 4)).repeat(repeats=(2, 1, 1))

    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(query_size=20, 
                                  key_size=2, 
                                  num_hiddens=8, 
                                  dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens).shape)