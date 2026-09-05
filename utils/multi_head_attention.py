import torch
import torch.nn as nn 
import numpy as np
from .self_attention import SelfAttention


class MutliHeadAttention(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()

        assert embed_dim % num_heads ==0, "embed_dim must be dicided by zero"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads



        # We'll use separate linear projections for each head, but we can also use
        # a single linear and reshape. For simplicity, we instantiate a SelfAttention
        # module for each head – not efficient but clear.
        self.heads = nn.ModuleList([
            SelfAttention(self.head_dim) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self,x:torch.Tensor,mask=None):
        batch, seq_len, _ = x.size()
        # split the last dimension into heads
        x_heads = x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # now (batch, num_heads, seq_len, head_dim)
        # apply each head separately (we loop for clarity)
        head_outputs = []
        for i, head in enumerate(self.heads):
            # input to each head: (batch, seq_len, head_dim)
            h_out, _ = head(x_heads[:, i, :, :])  # -> (batch, seq_len, head_dim)
            head_outputs.append(h_out)
        # concatenate heads along last dimension
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, embed_dim)
        out = self.out_proj(concat)
        return out


            