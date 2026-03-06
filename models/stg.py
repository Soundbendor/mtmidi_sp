from collections import OrderedDict
from torch import nn
import os
import torch
from torch.nn import functional as F

from .mlpprobe import MLPProbe
class StochasticGates(nn.Module):
    def __init__(self, in_dim =4096, sigma = 1, add_noise = True, hidden_dims = [], out_dim = 10, dropout = 0.5, initial_dropout = True):
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.sigma = sigma
        self.add_noise = add_noise
       
        # need k gumbel_softmax layers with dimension in_dim
        # to select k elements from in_dim elements
        self.probs = nn.Parameter(torch.ones(in_dim, dtype=self.ftype, requires_grad = True) * 0.5)
        self.classifier = MLPProbe(in_dim = k, hidden_dims = hidden_dims, out_dim = out_dim, dropout = dropout, initial_dropout = initial_dropout)
        
  
    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_add_noise(self,add_noise):
        self.add_noise = add_noise

    # to do
    def forward(self, x):
        feature_weights = F.gumbel_softmax(self.probs, tau = self.tau, hard = self.hard)
        weighted_input = x @ feature_weights.T # (batch_size, in_dim) @ (in_dim, k)

        out = self.classifier(weighted_input)
        return out
