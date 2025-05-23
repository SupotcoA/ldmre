import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.fc = nn.Linear(c_dim, 3 * n_channels, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, c=None):
        u = torch.mean(x, dim=1, keepdim=True)
        s = (x-u).pow(2).mean(1,keepdim=True)
        x = (x - u) / torch.sqrt(s + 1e-6)
        scale, bias, gamma = torch.chunk(self.fc(c), chunks=3, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias), gamma[:, :, None, None]
    


class AdaptiveGroupNorm(nn.Module):

    def __init__(self, n_groups, n_channels, c_dim):
        super().__init__()
        self.norm = nn.GroupNorm(n_groups, n_channels, affine=False)
        self.fc = nn.Linear(c_dim, 2 * n_channels, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, c=None):
        x = self.norm(x)
        scale, bias = torch.chunk(self.fc(c), chunks=2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        # return x.mul(1 + scale).add(bias)
        return torch.addcmul(bias, scale + 1, x)


class DyT(nn.Module):
    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.fc = nn.Sequential(nn.Linear(c_dim, 3 * n_channels, bias=True))

    def forward(self, x, c=None):
        x = torch.tanh(self.alpha * x)
        scale, bias, gamma = torch.chunk(self.fc(c), chunks=3, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias), gamma[:, :, None, None]
