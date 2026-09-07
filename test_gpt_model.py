import torch 
from gpt_model import GPT


vocab_size =1000
embed_dim,num_heads,num_layers,hidden_dim = 64,8,4,256
max_seq_len =128
batch,seq_len = 2,20

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = GPT(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_len)
model =model.to(device=device)
dummy_input = torch.randint(0, vocab_size, (batch, seq_len))
logits = model(dummy_input)
print("Logits shape:", logits.shape)