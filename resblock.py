import torch
from torch import nn
import torch.nn.functional as F
from norm import AdaptiveLayerNorm, DyT, AdaptiveGroupNorm


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

    def forward(self, x, c=None):
        h = self.conv(x)
        return h


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding='same')

    def forward(self, x, c=None):
        h = self.upsample(x)
        h = self.conv(h)
        return h





class ConvNeXtBlock(nn.Module):

    def __init__(self, 
                 in_dim,
                 cond_dim,
                 expansion_ratio=4
                ):
        super().__init__()

        def make_adn(*args,**kwargs):
            return AdaptiveLayerNorm(*args,**kwargs)

        self.conv1 = nn.Conv2d(in_channels=in_dim,
                               out_channels=in_dim,
                               kernel_size=7,
                               stride=1,
                               padding='same',
                               bias=True,
                               groups=in_dim)
        self.norm = make_adn(n_channels=in_dim,
                              c_dim=cond_dim) 
        mlp_dim = expansion_ratio * in_dim
        self.conv2 = nn.Conv2d(in_channels=in_dim,
                               out_channels=mlp_dim,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=mlp_dim,
                               out_channels=in_dim,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               bias=True)

    def forward(self, x, c=None):
        h = x
        h = self.conv1(h)
        h, gamma = self.norm(h, c)
        h = self.conv2(h)
        h = F.gelu(h)
        h = self.conv3(h)
        h = h * gamma
        return x + h
    

class ResBlock(nn.Module):
    def __init__(self, 
                 in_dim,
                 cond_dim,
                 out_dim=None,
                ):
        super().__init__()

        def make_adn(*args,**kwargs):
            return AdaptiveGroupNorm(*args,n_groups=16,**kwargs)

        out_dim = out_dim if out_dim is not None else in_dim
        self.norm1 = make_adn(n_channels=in_dim,
                              c_dim=cond_dim,)
        
        self.conv1 = nn.Conv2d(in_channels=in_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=True,
                               groups=1)
         
        self.norm2 = make_adn(n_channels=out_dim,
                              c_dim=cond_dim,) 
        
        self.conv2 = nn.Conv2d(in_channels=out_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=True)
        
        if out_dim==in_dim:
            self.conv_short_cut = nn.Identity()
        else:
            self.conv_short_cut = nn.Conv2d(in_channels=in_dim,
                                            out_channels=out_dim,
                                            kernel_size=1,
                                            stride=1,
                                            padding='same',
                                            bias=True)

    def forward(self, x, c=None):
        h = x
        h = self.norm1(h, c)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h, c)
        h = F.silu(h)
        h = self.conv2(h)
        return self.conv_short_cut(x) + h


class R3ResBlock(nn.Module):
    def __init__(self, 
                 in_dim,
                 cond_dim,
                 expansion_ratio=2,
                 out_dim=None
                ):
        super().__init__()

        def make_adn(*args,**kwargs):
            return AdaptiveGroupNorm(*args,n_groups=16,**kwargs)  # ??

        out_dim = out_dim if out_dim is not None else in_dim
        mid_dim = expansion_ratio * out_dim
        self.conv1 = nn.Conv2d(in_channels=in_dim,
                               out_channels=mid_dim,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               bias=True)
        self.norm1 = make_adn(n_channels=mid_dim,
                              c_dim=cond_dim) 
        
        self.conv2 = nn.Conv2d(in_channels=mid_dim,
                               out_channels=mid_dim,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=True,
                               groups=max(1,mid_dim//16))
        self.norm2 = make_adn(n_channels=mid_dim,
                              c_dim=cond_dim) 
        
        self.conv3 = nn.Conv2d(in_channels=mid_dim,
                               out_channels=out_dim,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               bias=False)
        
        if out_dim==in_dim:
            self.conv_short_cut = nn.Identity()
        else:
            self.conv_short_cut = nn.Conv2d(in_channels=in_dim,
                                            out_channels=out_dim,
                                            kernel_size=1,
                                            stride=1,
                                            padding='same',
                                            bias=True)

    def forward(self, x, c=None):
        h = x
        h = self.conv1(h)
        h = self.norm1(h, c)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h, c)
        h = F.silu(h)
        h = self.conv3(h)
        return self.conv_short_cut(x) + h