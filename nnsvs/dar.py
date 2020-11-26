# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu

class MDNDARCell(nn.Module):
    """ Cell of Deep Autoregressive model with Mixture Density Network.

    Attributes:
        in_dim (int): the number of dimensions of the input
        hidden_dim(int): the number of dimensions of the hidden layer
        out_dim (int): the number of dimensions of the output
        num_gaussians (int): the number of Gaussians component
    """    
    def __init__(self, in_dim, hidden_dim, out_dim, num_gaussians=8, dim_wise=False):
        super(MDNDARCell, self).__init__()
        
        #B, D_in
        self.rnncell = nn.RNNCell(in_dim + out_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.mdnlayer = MDNLayer(hidden_dim, out_dim, num_gaussians, dim_wise)
        
    def forward(self, x, hidden):
        h = self.rnncell(x, hidden)
        out = self.linear(h)

        # B, hidden_dim -> B, 1, hidden_dim
        out = out.unsqueeze(1)
        
        log_pi, log_sigma, mu = self.mdnlayer(out)
        return log_pi, log_sigma, mu, h
        
class MDNDAR(nn.Module):
    """ Deep Autoregressive model with Mixture Density Networ

    The input maps to the parameters of a Mixture of Gaussians (MoG) probability
    distribution, where each Gaussian has out_dim dimensions and diagonal covariance.
    If dim_wise is True, features for each dimension are modeld by independent 1-D GMMs
    instead of modeling jointly. This would workaround training difficulty
    especially for high dimensional data.

    The previous mu(mean) of MDNDARCell is concatinated with the input and re-used.

    Implementation reference:
        1. https://ieeexplore.ieee.org/document/8341752

    Attributes:
        in_dim (int): the number of dimensions of the input
        hidden_dim(int): the number of dimensions of the hidden layer
        out_dim (int): the number of dimensions of the output
        dropout (float): dropout ratio of auto-regressed value
        num_gaussians (int): the number of Gaussians component
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2, num_gaussians=8, dim_wise=False):
        super(MDNDAR, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_gaussians=num_gaussians
        self.dropout = nn.Dropout(dropout)
        self.mdndarcell = MDNDARCell(in_dim, hidden_dim, out_dim, num_gaussians, dim_wise)

    def forward(self, x, lengths=None):
        """Forward for MDNDAR

        Args:
            minibatch (torch.Tensor): tensor of shape (B, T, D_in)
                B is the batch size and T is data lengths of this batch,
                and D_in is in_dim.

        Returns:
            torch.Tensor: Tensor of shape (B, T, G) or (B, T, G, D_out)
                Log of mixture weights. G is num_gaussians and D_out is out_dim.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                the log of standard deviation of each Gaussians.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                mean of each Gaussians
        """
        
        B, T, _ = x.shape
        hidden = torch.zeros(B, self.hidden_dim, device=x.device)
        
        log_pi = torch.Tensor().to(x.device)
        log_sigma = torch.Tensor().to(x.device)
        mu = torch.Tensor().to(x.device)
        
        for idx in range(T):
            if idx == 0:
                inputs = torch.cat((x[:,idx,:], torch.zeros(B, self.out_dim, device=x.device)), dim=1)
            else:
                if len(log_pi.shape) == 4:
                    # (B, 1, G, D_out), (B, 1, G, D_out), (B, 1, G, D_out) -> (B, 1, D_out)
                    _, prev_mu = mdn_get_most_probable_sigma_and_mu(log_pi[:,idx-1:idx,:,:], log_sigma[:,idx-1:idx,:,:], mu[:,idx-1:idx,:,:])                    
                else:
                    # (B, 1, G), (B, 1, G, D_out), (B, 1, G, D_out) -> (B, 1, D_out)
                    _, prev_mu = mdn_get_most_probable_sigma_and_mu(log_pi[:,idx-1:idx,:], log_sigma[:,idx-1:idx,:,:], mu[:,idx-1:idx,:,:])                    
                    # B, 1, D_out -> B, D_out
                prev_mu = prev_mu.squeeze(1)
                inputs = torch.cat((x[:,idx,:], self.dropout(prev_mu)), dim=1)
                
            _lp, _ls, _m, hidden = self.mdndarcell(inputs, hidden)
            
            log_pi = torch.cat((log_pi, _lp), dim=1)
            log_sigma = torch.cat((log_sigma, _ls), dim=1)
            mu = torch.cat((mu, _m), dim=1)
                                           
        return log_pi, log_sigma, mu
