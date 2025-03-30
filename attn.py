import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttnBlock(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads=8):
        super().__init__()
        self.channels = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.attention = nn.MultiheadAttention(embed_dim=dim, 
                                               num_heads=num_heads, 
                                               batch_first=True)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        # inp [B,C,H,W]
        b, c, h, w = x.shape
        
        # Reshape and permute to [B, H*W, C]
        x_flat = x.reshape(b, c, -1).transpose(1, 2).contiguous()
        
        # Apply layer norm
        normed = self.norm(x_flat)
        
        # Generate q,k,v
        qkv = self.to_qkv(normed)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        
        # Apply attention
        attn_out, _ = self.attention(q, k, v)
        
        # Project and reshape back
        out = self.proj_out(attn_out)
        out = out.transpose(1, 2).contiguous().reshape(b, c, h, w)
        
        # Residual connection
        return x + out