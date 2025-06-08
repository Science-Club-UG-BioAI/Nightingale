import torch
import torch.nn as nn
from embedding import Amino_embeder
from encoder import EncoderTransformer
from decoder import Decoder


class MonMothmaTheGOAT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, neuron, twink, mushroom, cla, dropout=0):
        super().__init__()
        
        self.src_embed = Amino_embeder(vocab_size, d_model)
        self.tgt_emved = Amino_embeder(vocab_size, d_model)
        
        self.encoder_layers = nn.ModuleList([
            EncoderTransformer(d_model, nhead, neuron, dropout)
            for _ in range(twink)])
        self.decoder_layers = nn.ModuleList([
            Decoder(d_model, nhead, neuron, dropout)
            for _ in range(mushroom)])
        self.output_layer = nn.Linear(d_model, cla)
        
    def forward(self, src, tgt, tgt_mask=None):
        src = self.src_embed(src)
        tgt = self.tgt_emved(tgt)
        
        for layer in self.encoder_layers:
            src = layer(src)
        memory = src
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask)
        
        return self.output_layer(tgt)
        
        




















