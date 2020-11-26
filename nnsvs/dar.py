# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from nnsvs.mdn import MDNLayer

class MDNDARCell(nn.Module):
    """ Cell of Deep Autoregressive model with Mixture Density Network.

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
        h = self.rnncell(x, hidden)
        out = nn.Linear(h)
        print(f"out.shape: {out.shape}")

        # B, hidden_dim -> B, 1, hidden_dim
        print(f"out.shape: {out.shape}")
        out = out.unsqueeze(1)
        print(f"out.shape: {out.shape}")
        out = self.mdnlayer(out)
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

    def forward(self, x, lengths):

        print(f"x.shape: {x.shape}")
        
        B, T, _ = x.shape
        hidden = torch.zeros(B, self.hidden_dim, device=x.device)
        
        log_pi = torch.Tensor()
        log_sigma = torch.Tensor()
        mu = torch.Tensor()
        
        for idx in range(T):
            if idx == 0:
                inputs = torch.cat((x[:,idx,:], torch.zeros(B, self.out_dim, device=x.device)), dim=1)
            else:
                inputs = torch.cat((x[:,idx,:], self.dropout(mu[:, idx-1, :])), dim=1)
            _lp, _ls, _m, hidden = self.mdndarcell(inputs, hidden)
            print(f"_lp.shape: {_lp.shape}")
            
            log_pi = torch.cat((log_pi, _lp), dim=1)
            log_sigma = torch.cat((log_sigma, _ls), dim=1)
            mu = torch.cat((mu, _m), dim=1)

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
