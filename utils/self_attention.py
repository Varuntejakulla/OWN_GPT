import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)

        # --- Add causal mask (no peeking into the future) ---
        seq_len = x.size(1)
        # Create a lower‑triangular matrix of ones: (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # Expand to (1, 1, seq_len, seq_len) for broadcasting
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        # Add a head dimension to scores: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        scores = scores.unsqueeze(1)
        # Mask out (set to -inf) all positions where causal_mask == 0 (future tokens)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        scores = scores.squeeze(1)  # back to (batch, seq_len, seq_len)
        # ----------------------------------------------------

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out, attn_weights