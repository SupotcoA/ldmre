import numpy as np
import torch
from torch import nn

class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, n_patch, emb_dim):
        super().__init__()
        self.n_patch = n_patch  # number of patches in a column/row
        self.emb_dim = emb_dim
        self.pos_emb = nn.Parameter(torch.zeros(n_patch * n_patch, emb_dim))
    
    def forward(self, x=None):
        return self.pos_emb


class TimeEmbedding(nn.Module):
    def __init__(self, t_max, cond_dim):
        super().__init__()
        self.time_emb = nn.Parameter(self.get_time_emb(t_max, cond_dim), 
                                     requires_grad=False)
    
    def get_time_emb(self, t_max, cond_dim):
        # sinusoidal time embedding
        timesteps = torch.arange(1, 1 + t_max)
        half_dim = cond_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if cond_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
    
    def forward(self, t):
        # 1<=t<=t_max
        return self.time_emb[t-1]


class ClassEmbedding(nn.Module):
    def __init__(self, n_class, cond_dim):
        super().__init__()
        self.cls_emb = nn.Parameter(torch.zeros(n_class, cond_dim))
    
    def forward(self, cls):
        return self.cls_emb[cls]
    

class FourierNoiseEmbedding(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(cond_dim).view(1,-1), 
                                 requires_grad=False)
        self.phi = nn.Parameter(torch.rand(cond_dim), 
                                 requires_grad=False)
    
    def forward(self, a):
        # a = ln(sigma)/4
        # [B,]
        return torch.cos(2*torch.pi*(torch.matmul(a.view(-1,1),self.freq)+self.phi))