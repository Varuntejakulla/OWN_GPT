import torch
import torch.nn as nn 
import torch.nn.functional as  F 



class SelfAttention(nn.Module):

    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(in_features=embed_dim,out_features=embed_dim)
        self.key   = nn.Linear(in_features=embed_dim,out_features=embed_dim)
        self.value = nn.Linear(in_features=embed_dim,out_features=embed_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out, attn_weights


    