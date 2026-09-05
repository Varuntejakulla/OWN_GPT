import torch
import math


def positional_encoding(seq_len,embed_dim,device="cpu"):

    pe= torch.zeros(seq_len,embed_dim,device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() *
                         (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


