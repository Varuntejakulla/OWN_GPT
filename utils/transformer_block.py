import torch.nn as nn 
from .multi_head_attention import MutliHeadAttention
from .feed_forward import FeedForward
import torch

class TransfromerBlock(nn.Module):


    def __init__(self, embed_dim,num_heads,hidden_dim,dropout=0.1):
        super().__init__()
        self.attn = MutliHeadAttention(embed_dim,num_heads)
        self.ffn = FeedForward(embed_dim,hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x
    