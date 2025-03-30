import numpy as np
import torch
from torch import nn
from embedding import TimeEmbedding,LearnablePositionalEncoding2D,ClassEmbedding

# this file implements modules for diffusion transformer
# use adaptive layer norm & classifier-free guidance 
# for class conditioned generation

# x: input tensor, shape (batch, n_patch * n_patch, emb_dim)
# t: time index, shape (batch,)
# cls: class index, shape (batch,)
# pos_emb: positional embedding, shape (n_patch * n_patch, emb_dim)




class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 cond_dim,
                 num_heads,
                 mlp_dim,
                 dropout,
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.cond_emb = nn.Linear(cond_dim, 6 * emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(emb_dim,elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
        )

    def forward(self, x, cond):
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.cond_emb(cond).chunk(6, dim=-1)
        x_ = self.norm1(x) * gamma1 + beta1
        x = x + self.attn(x_, x_, x_)[0] * alpha1
        x_ = self.norm2(x) * gamma2 + beta2
        x = x + self.mlp(x_) * alpha2
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 inp_dim,
                 emb_dim,
                 cond_dim,
                 num_heads,
                 mlp_dim,
                 num_layers,
                 dropout,
                 **kwargs
                 ):
        super().__init__()
        self.proj_in = nn.Linear(inp_dim, emb_dim)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, cond_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(emb_dim, inp_dim)
    
    def forward(self, x, pos_emb, cond):
        x = self.proj_in(x)
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x, cond)
        x = self.proj_out(x)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 transformer_config):
        super().__init__()
        self.encoder = TransformerEncoder(**transformer_config)
        self.t_emd = TimeEmbedding(transformer_config['t_max'],
                                    transformer_config['cond_dim'])
        self.pos_emb = LearnablePositionalEncoding2D(transformer_config['n_patch'],
                                                     transformer_config['emb_dim'])
        self.cls_emb = ClassEmbedding(transformer_config['n_class'],
                                      transformer_config['cond_dim'])
    
    def forward(self, x, t, cls, mask=None):
        t_emb = self.t_emd(t)
        cls_emb = self.cls_emb(cls)
        if mask is not None:
            cls_emb[mask] = 0
        pos_emb = self.pos_emb()
        return self.encoder(x, pos_emb, t_emb + cls_emb)

