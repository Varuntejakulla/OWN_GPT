import torch 
from utils.tokenizer import CharTokenizer
from gpt_model import GPT
import os 

datset_path ="./Data/trainig_datatset.txt"
with open(datset_path,'r',encoding='utf-8')as file :
    text = file.read()

tokenizer = CharTokenizer(text=text)
vocab_size = tokenizer.vocab_size

# Hyperparameters (must match training)

embed_dim = 128
num_heads = 8
num_layers = 4
hidden_dim = 512
max_seq_len = 64


device = torch.device('cuda'if  torch.cuda.is_available() else 'cpu')

model = GPT(vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            ).to(device=device)
model.load_state_dict(torch.load('./own_gpt.pt',map_location=device))
print("model load successfully")
eval =model.eval()
print(eval)

def generate(prompt,max_new_tokens=100,temperature=1.0):
     
    input_ids= tokenizer.encode_tensor(prompt).unsqueeze(0)
    for _ in range(max_new_tokens):
        if input_ids.size(1)>max_seq_len:
            input_ids= input_ids[:,-max_seq_len:]
        with torch.no_grad():
            logits = model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    generated:torch.Tensor = tokenizer.decode(input_ids.squeeze(0).tolist())
    return generated

prompt = "Who Stole the Tarts "
print(generate(prompt=prompt,max_new_tokens=200,temperature=1.2))

