# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-11-28
# @author: cnzero

"""hold on, deeply think,
and add some document descriptions for the script.
"""

import torch
from torch import nn

class Decoder(nn.Module):
    """the base decoder interface for the encoder-decoder architecture.
    """
    def __init__(self):
        super().__init__()

    def init_state(self, 
                   enc_outputs):
        raise NotImplementedError
    
    def forward(self,
                X, 
                state):
        raise NotImplementedError

class AttentionDecoder(Decoder):
    
