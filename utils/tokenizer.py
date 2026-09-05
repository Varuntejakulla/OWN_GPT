import torch 

class CharTokenizer:

    def __init__(self,text,special_tokens=None):

        chars =sorted(list(set(text)))
        if special_tokens:
            chars= special_tokens+chars
            self.stoi = {ch:i for i,ch in enumerate(chars)}
            self.itos = {i:ch for i,ch in enumerate(chars)}
            self.vocab_size = len(chars)

    def encode(self,text):

        return[self.stoi[ch] for ch in text]


    def decode(self,ids):
        return ''.join([self.itos[i]for i in ids])


    def encode_tensor(self,text,device="cpu"):
        return torch.tensor(
            self.encode(text),
            dtype=torch.long,
            device=device
        )


     
        