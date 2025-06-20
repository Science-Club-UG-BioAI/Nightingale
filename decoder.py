from torch import nn
import torch
from typing import Optional

class Decoder(nn.Module): #decoder class, which inherits from a nn.Module
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout=0.1): #decoder class constructor: (d_model-embeddings size; nhead-attention heads; dim_forward-number of neurons in frist linear layer)
        super().__init__() #superclass constructor
        
        self.m_attention = nn.MultiheadAttention( #masked self-attention -> attends to previous decoder tokens
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.m_cross_attention = nn.MultiheadAttention( #cross-attention -> attends to encoder output
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.sequential = nn.Sequential( #feed-forward network
            nn.Linear(d_model, dim_feedforward), #linear layer
            nn.ReLU(), #activation function ReLU
            nn.Dropout(dropout), #dropout layer to prevent overfitting
            nn.Linear(dim_feedforward, d_model) #linear layer
        )
        #3 normalization layers to stabilize training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) #dropout layer to prevent overfitting
    
    def forward(self, x: torch.Tensor, memory:torch.Tensor, tgt_mask: Optional[torch.Tensor] = None): #forward function (x-data generated by decoder; memory-data generated by encoder, tgt_mask-prevent attention to future positions)
        _x, _ = self.m_attention(x, x, x, attn_mask=tgt_mask) #passing data through multihead attention layer
        x = self.norm1(x + self.dropout(_x)) #passing data through dropout and normalization layer and adding the residual to stabilize training
        _x, _ = self.m_cross_attention(x, memory, memory) #passing data through multihead cross layer
        x = self.norm2(x+self.dropout(_x)) #passing data through droput layer and adding the residual to stabilize training
        _x = self.sequential(x) #passing data through feed-forward network
        x = self.norm3(x + self.dropout(_x)) #passing data through droput layer and adding the residual to stabilize training

        return x #returning output
