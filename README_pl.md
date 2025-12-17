# Projekt Słowik

Mały projekt w PyTorch: 
własna implementacja Transformera encoder–decoder do pracy na sekwencjach (tokeny aminokwasów $\to$  tokeny opisu). 

English version [here](README.md)

## Pliki

* `Venator.py` - główny model, łączy ze soba enkoder i dekoder
* `embedding.py` - embedding + sinusoidalny positional encoding
* `encoder.py` - blok encodera
* `decoder.py` - blok decodera

## Wymagania
* [Python 3.9+](https://www.python.org/)
* [Pytorch](https://pytorch.org/)

## Instalacja bibliotek

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```

## Przykładowe użycie - forward pass
### Przygotowanie
1. Utwórz plik:
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
        twink=4,        # liczba warstw encodera
        mushroom=4,     # liczba warstw decodera
        cla=128,        # rozmiar słownika wyjściowego
        dropout=0.1
    )

    src = torch.randint(0, 32, (B, S))
    tgt = torch.randint(0, 128, (B, T))
    tgt_mask = subsequent_mask(T)

    logits = model(src, tgt, tgt_mask=tgt_mask)
    print("logits:", logits.shape)
    ```
2. Uruchom skrypt w swoim środowisku:
    ```bash
    python run_demo.py
    ```
### Wyjscie
`logits` o kształcie (batch, tgt_len, vocab_out), czyli rozkład po tokenach wyjściowych dla każdej pozycji dekodera.

## Uwagi
To jest “rdzeń modelu”. Trening (loss, teacher forcing, PAD/ignore_index, beam search) trzeba dopisać osobno.



