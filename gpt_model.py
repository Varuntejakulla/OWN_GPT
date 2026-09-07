import torch 
import torch.nn as nn
from utils.embeddings import TokenEmbedding
from utils.transformer_block import TransfromerBlock
from  utils.postional_encoding import positional_encoding


class GPT(nn.Module):

    def __init__(self,
                vocab_size, 
                embed_dim, 
                num_heads, 
                num_layers, 
                hidden_dim,
                max_seq_len,
                dropout=0.1
                ):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size,embed_dim)
        self.pos_enc  = positional_encoding(max_seq_len,embed_dim)
        self.blocks = nn.ModuleList([
            TransfromerBlock(embed_dim,num_heads,hidden_dim,dropout)
            for _ in range(num_layers)

        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim,vocab_size)
        self.max_seq_len = max_seq_len
        mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len
            )
        )

        self.register_buffer(
            "causal_mask",
            mask
        )

        

    def forward(self, idx, mask=None):
        # idx: (batch, seq_len)
        batch, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # token embeddings
        x = self.token_embed(idx)  # (batch, seq_len, embed_dim)
        # add positional encoding (slice to match seq_len)
        x = x + self.pos_enc[:seq_len, :].unsqueeze(0).to(x.device)
        mask = self.causal_mask[
            :seq_len,
            :seq_len
        ]
       

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_final(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        return logits