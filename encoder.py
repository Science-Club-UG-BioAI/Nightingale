import torch
from torch import nn
import torch.nn as nn
import math

class EncoderTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # Validate input types
        if not isinstance(d_model, int):
            raise TypeError(f"d_model must be int, got {type(d_model).__name__}")
        if not isinstance(nhead, int):
            raise TypeError(f"nhead must be int, got {type(nhead).__name__}")
        if not isinstance(dim_feedforward, int):
            raise TypeError(f"dim_feedforward must be int, got {type(dim_feedforward).__name__}")
        if not isinstance(dropout, float):
            raise TypeError(f"dropout must be float, got {type(dropout).__name__}")

        # Multi-head self-attention mechanism
        self.multiheadAttention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Position-wise feedforward neural network
        self.sequential = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),  # Project to higher dimension
            nn.ReLU(),                            # Apply non-linearity
            nn.Dropout(dropout),                  # Regularization
            nn.Linear(dim_feedforward, d_model)   # Project back to original dimension
        )

        # Layer normalization for stabilizing training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply multi-head attention (self-attention)
        _x, _ = self.multiheadAttention(x, x, x)
        # Add & Norm (residual connection + layer normalization)
        x = self.norm1(x + self.dropout(_x))

        # Apply feedforward network
        _x = self.sequential(x)
        # Add & Norm (residual connection + layer normalization)
        x = self.norm2(x + self.dropout(_x))
        
        return x
#positional encoding moved to embedding.py
