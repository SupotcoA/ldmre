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
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True
        )

    def forward(self, x):
        # inp [B,C,H,W]
        b, c, h, w = x.shape
        
        # Reshape and permute to [B, H*W, C]
        x_flat = x.reshape(b, c, -1).transpose(1, 2).contiguous()
        
        # Apply layer norm
        normed = self.norm(x_flat)
        
        # Apply attention (nn.MultiheadAttention handles QKV internally)
        attn_out = self.attention(normed, normed, normed,
                                  need_weights=False)[0]
        
        # Reshape back
        out = attn_out.transpose(1, 2).contiguous().reshape(b, c, h, w)
        
        # Residual connection
        return x + out