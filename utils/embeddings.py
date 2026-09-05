import torch 
import torch.nn as  nn
import math 


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size,embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim)


    def forward(self,x):

        return self.embed(x)

    