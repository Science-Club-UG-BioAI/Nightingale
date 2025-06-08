import torch
from torch import nn
import torch.nn as nn
import math

class EncoderTransformer(nn.Module):
  def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
    if not isinstance(d_model, int):
        raise TypeError(f"d_model must be int, got {type(d_model).__name__}")
    if not isinstance(nhead, int):
        raise TypeError(f"nhead must be int, got {type(nhead).__name__}")
    if not isinstance(dim_feedforward, int):
        raise TypeError(f"dim_feedforward must be int, got {type(dim_feedforward).__name__}")
    if not isinstance(dropout, float):
        raise TypeError(f"dropout must be float, got {type(dropout).__name__}")
    self.multiheadAttention = nn.MultiheadAttention(
        d_model,
        nhead,
        dropout=dropout,
        batch_first=True
    )
    self.sequential = nn.Sequential(
        nn.Linear(
          d_model,
          dim_feedforward
        ),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(
          dim_feedforward,
          d_model
        )
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    x = self.multiheadAttention(x)
    x = self.dropout(x)
    x = self.norm1(x)
    x = self.sequential(x)
    x = self.dropout(x)
    x = self.norm2(x)
    return x

#positional encoding moved to embedding.py
