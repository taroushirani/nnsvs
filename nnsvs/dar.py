# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from nnsvs.mdn import MDNLayer

class MDNDARCell(nn.Module):
    """ Cell of Deep Autoregressive model with Mixture Density Networ.

    Attributes:
        in_dim (int): the number of dimensions of the input
        hidden_dim(int): the number of dimensions of the hidden layer
        out_dim (int): the number of dimensions of the output
        num_gaussians (int): the number of Gaussians component
    """    
    def __init__(self, in_dim, hidden_dim, out_dim, num_gaussians=8):
        super(MDNDARCell, self).__init__()
        
        #B, D_in
        self.rnncell = nn.RNNCell(in_dim + out_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.mdnlayer = MDNLayer(hidden_dim, out_dim, num_gaussians=num_gaussians)
        
    def forward(self, x, hidden):
        print(f"x.shape: {x.shape}")
        out, h = self.rnncell(x, hidden)
        out = self.mdnlayer(self.linear(out))
        return out, h
        
class MDNDAR(nn.Module):
    """ Deep Autoregressive model with Mixture Density Networ

    The input maps to the parameters of a Mixture of Gaussians (MoG) probability
    distribution, where each Gaussian has out_dim dimensions and diagonal covariance.

    Attributes:
        in_dim (int): the number of dimensions of the input
        hidden_dim(int): the number of dimensions of the hidden layer
        out_dim (int): the number of dimensions of the output
        dropout (float): dropout ratio of auto-regressed value
        num_gaussians (int): the number of Gaussians component
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2, num_gaussians=8):
        super(MDNDAR, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_gaussians=num_gaussians
        self.dropout = nn.Dropout(dropout)
        self.mdndarcell = MDNDARCell(in_dim, hidden_dim, out_dim, num_gaussians)

    def forward(self, x, length):

        print(f"x.shape: {x.shape}")
        
        B, T, _ = x.shape
        hidden = torchs.zeros(self.hidden_dim)
        
        log_pi = torch.Tensor()
        log_sigma = torch.Tensor()
        mu = torch.Tensor()
        
        for idx in range(T):
            if idx == 1:
                inputs = torch.cat(sequence[idx], torch.zeros(self.out_dim))
            else:
                inputs = torch.cat(sequence[idx], self.dropout(mu[idx-1]))
                _lp, _ls, _m, hidden = self.mdndarcell(inputs, hidden)
                log_pi = torch.cat((log_pi, _lp))
                log_sigma = torch.cat((log_sigma, _ls))
                mu = torch.cat((mu, _m))

        # B, T, G
        print(f"log_pi.shape: {log_pi.shape}")
        log_pi = log_pi.view(B, T, self.out_dim)
        print(f"log_pi.shape: {log_pi.shape}")
        
        # B, T, G, D_out
        print(f"log_sigma.shape: {log_sigma.shape}")        
        log_sigma = log_pi.view(B, T, self.num_gaussians, self.out_dim)
        print(f"log_sigma.shape: {log_sigma.shape}")        
        
        # B, T, G, D_out
        print(f"log_mu.shape: {log_mu.shape}")                
        mu = mu.view(B, T, self.num_gaussians, self.out_dim)
        print(f"log_mu.shape: {log_mu.shape}")                
        
        return log_pi, log_sigma, mu
