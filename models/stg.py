from collections import OrderedDict
from torch import nn
import os
import torch
from torch.nn import functional as F
import torch.distributions.normal as TDN

from .mlpprobe import MLPProbe
class StochasticGates(nn.Module):
    def __init__(self, in_dim =4096, sigma = 0.5, add_noise = True, hidden_dims = [], out_dim = 10, dropout = 0.5, initial_dropout = True, generator = None):
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.sigma = sigma
        # to add noise or not
        self.add_noise = add_noise
        self.generator = generator

        # need k gumbel_softmax layers with dimension in_dim
        # to select k elements from in_dim elements
        self.means = nn.Parameter(torch.ones(in_dim, dtype=self.ftype, requires_grad = True) * 0.5)
        self.classifier = MLPProbe(in_dim = k, hidden_dims = hidden_dims, out_dim = out_dim, dropout = dropout, initial_dropout = initial_dropout)
        
  
    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_add_noise(self,add_noise):
        self.add_noise = add_noise

    def get_sum_cdf(self):
        return torch.sum(TDN.Normal(0,1).cdf(self.means/self.sigma))

    def forward(self, x):
        cur_gate_val = self.means
        if self.add_noise == True:
            # if add noise, add gaussian noise drawn from N(0,sigma^2)
            cur_gate_val += (torch.randn_like(self.means, generator = self.generator) * self.sigma)
        clamped_gate_val = torch.clamp(cur_gate_val)
        
        weighted_input = torch.mul(x, clamped_gate_val)

        out = self.classifier(weighted_input)
        return out
