from torch import nn
import torch
from typing import Optional

class Decoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout=0.1):
        super().__init__()
        
        self.m_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.m_cross_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.sequential = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, memory:torch.Tensor, tgt_mask: Optional[torch.Tensor] = None):
        _x, _ = self.m_attention(x, x, x, tgt_mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))
        _x, _ = self.m_cross_attention(x, memory, memory)
        x = self.norm2(x+self.dropout(_x))
        _x = self.sequential(x)
        x = self.norm3(x + self.dropout(_x))

        return x
