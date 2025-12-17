# Nightingale
A small PyTorch project:
a custom implementation of a Transformer encoder–decoder for sequence modeling (amino acid tokens $\to$ description tokens).

Polska wersja [tutaj](README_pl.md).

## Files
* `Venator.py` - main model that connects the encoder and decoder
* `embedding.py` - embedding + sinusoidal positional encoding
* `encoder.py` - encoder block
* `decoder.py` - decoder block

## Requirements
* [Python 3.9+](https://www.python.org/)
* [PyTorch](https://pytorch.org/)

## Installing dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```

## Example usage: forward pass
1. Setup
Create a file:
    ```python
    import torch
    from Venator import MonMothmaTheGOAT

    def subsequent_mask(T: int, device=None):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    B, S, T = 2, 32, 16

    model = MonMothmaTheGOAT(
        vocab_size=32,
        d_model=256,
        nhead=8,
        neuron=1024,
        twink=4,        # number of encoder layers
        mushroom=4,     # number of decoder layers
        cla=128,        # output vocabulary size
        dropout=0.1
    )

    src = torch.randint(0, 32, (B, S))
    tgt = torch.randint(0, 128, (B, T))
    tgt_mask = subsequent_mask(T)

    logits = model(src, tgt, tgt_mask=tgt_mask)
    print("logits:", logits.shape)
    ```
2. Run the script in your environment:

    ```bash
    python run_demo.py
    ```

### Output
`logits` has shape (batch, tgt_len, vocab_out), meaning a distribution over output tokens for each decoder position.

## Notes
This is the “core model”. Training (loss, teacher forcing, PAD/ignore_index, beam search) needs to be implemented separately.