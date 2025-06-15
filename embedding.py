import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module): 
    def __init__(self, d_model: int, max_len=800):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("pe", pe) 

    def forward(self, x: torch.tensor):
        return x + self.pe[:x.size(1)]



class Amino_embeder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        
    def forward(self,x):
        return self.pe(self.embedding(x))
    











