import torch 
import torch.nn as nn 
from  torch.utils.data import DataLoader, Dataset
from  utils.tokenizer import CharTokenizer
from gpt_model import GPT
import os 


embed_dim =128
num_heads = 8
num_layers = 4
hidden_dim = 512
max_seq_len =64
batch_size = 32
epochs = 100
learning_rate = 3e-4
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

datset_path ="./Data/trainig_datatset.txt"
with open(datset_path,'r',encoding='utf-8')as file :
    traing_data = file.read()

tokenizer = CharTokenizer(text=traing_data)
print(vars(tokenizer))
vocab_size = tokenizer.vocab_size

#preparing the datset:chunks text and max  sequence length 
data = torch.tensor(tokenizer.encode(text=traing_data),dtype=torch.long)
n= len(data)

train_data = data[: n-(n%(max_seq_len+1))]
train_data =train_data.view(-1,max_seq_len+1)
# Each sequence: x = seq[:-1], y = seq[1:]

class TetxtDataset(Dataset):
    def __init__(self,data):
        self.data =data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq =self.data[idx]
        x=seq[:-1]
        y=seq[1:]
        return x,y

data_set = TetxtDataset(train_data)
dataloader= DataLoader(dataset=data_set,batch_size=batch_size,shuffle=True)


model = GPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    max_seq_len=max_seq_len
).to(device=device)

#An optimizer is used during training to update the model's weights so that the model makes fewer mistakes
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in  range(epochs):
    total_loss = 0
    for x,y in dataloader:
        x:torch.Tensor = x.to(device)
        y:torch.Tensor =y.to(device)
        # Logits are the raw scores that your neural network produces before converting them into probabilities.
        logits:torch.Tensor= model(x)
         # reshape for cross entropy: (batch*seq_len, vocab_size) vs (batch*seq_len)
        loss:torch.Tensor= criterion(logits.view(-1,vocab_size),y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    avg_loss = total_loss/len(dataloader)

    if epoch%10 ==0:
        print(f"Epoch {epoch}, Loss :{avg_loss:.4f}")

torch.save(model.state_dict(),"own_gpt.pt")
print("Model saved successfully ")





