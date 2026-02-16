from collections import OrderedDict
from torch import nn
import os
import torch

class LinearNNProbe(nn.Module):
    def __init__(self, in_dim =4096, out_dim = 10, dropout = 0.5, initial_dropout = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.initial_dropout = True
        
        cur_layers = []

        if initial_dropout == True:
            cur_layers.append( ('dropout', nn.Dropout(p=dropout) ))
        cur_layers.append( ('linear', nn.Linear(in_dim, out_dim)) )
        self.layers = nn.Sequential(OrderedDict(cur_layers))

    def forward(self, x):
        out = self.layers(x)
        return out
