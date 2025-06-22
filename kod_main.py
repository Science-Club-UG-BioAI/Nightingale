import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import sacrebleu

def encode(seq, vocab):
    return [vocab.get(x, vocab['<pad>']) for x in seq]

def build_vocab(tokens):
    vocab = {tok: i+1 for i, tok in enumerate(sorted(set(tokens)))}
    vocab['<pad>'] = 0
    return vocab

class ProteinDataset(Dataset):
    def __init__(self, sequences, descriptions, src_vocab, tgt_vocab):
        self.sequences = sequences
        self.descriptions = descriptions
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        src = torch.tensor(encode(self.sequences[idx], self.src_vocab), dtype=torch.long)
        tgt_seq = ['<start>'] + self.descriptions[idx] + ['<end>']
        tgt = torch.tensor(encode(tgt_seq, self.tgt_vocab), dtype=torch.long)
        return src, tgt, tgt_seq

def collate_fn(batch):
    src, tgt, desc = zip(*batch)
    return pad_sequence(src, True, 0), pad_sequence(tgt, True, 0), desc

class ProteinModel(pl.LightningModule):
    def __init__(self, sequences, descriptions, sequences_val, descriptions_val, 
                 src_vocab, tgt_vocab, src_vocab_size, tgt_vocab_size, d_model, nhead, 
                 neuron, num_encoder_layers, num_decoder_layers, dropout=0.2, 
                 lr=0.0001, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Transformer(d_model=d_model, nhead=nhead,
                                    num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=neuron,
                                    dropout=dropout, batch_first=True)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
        self.validation_outputs = []

    def forward(self, src, tgt, tgt_mask=None):
        return self.fc_out(self.model(self.src_embedding(src), self.tgt_embedding(tgt), tgt_mask=tgt_mask))

    def shared_step(self, batch, is_val=False):
        src, tgt, desc = batch
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
        mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)
        logits = self(src, tgt_input, mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        pred = logits.argmax(dim=-1)
        acc = ((pred == tgt_output) & (tgt_output != 0)).sum().float() / (tgt_output != 0).sum().float()
        out = {"loss": loss, "acc": acc}
        if is_val:
            texts = [" ".join(self.inv_tgt_vocab.get(i.item(), "<unk>") for i in row if i != 0) for row in pred]
            refs = [" ".join(d[1:-1]) for d in desc]
            out.update({"predicted_texts": texts, "references": refs})
        return out

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", out["acc"], on_step=True, on_epoch=True, prog_bar=True)
        return out

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, is_val=True)
        self.validation_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        vals = self.validation_outputs
        val_loss = torch.stack([x["loss"] for x in vals]).mean()
        val_acc = torch.stack([x["acc"] for x in vals]).mean()
        preds = [t for x in vals for t in x["predicted_texts"]]
        refs = [r for x in vals for r in x["references"]]
        bleu = sacrebleu.corpus_bleu(preds, [refs]).score
        self.log("val_bleu", bleu, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True)
        print(f"Epoch {self.current_epoch}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, BLEU {bleu:.4f}")
        
        # Dodane: zapisywanie modelu po każdej epoce
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_bleu': bleu
        }, f'model_epoch.pt')
        
        self.validation_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        return {"optimizer": opt, "lr_scheduler": sched}

    def train_dataloader(self):
        return DataLoader(ProteinDataset(self.hparams.sequences, self.hparams.descriptions,
                                         self.hparams.src_vocab, self.hparams.tgt_vocab),
                          batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(ProteinDataset(self.hparams.sequences_val, self.hparams.descriptions_val,
                                         self.hparams.src_vocab, self.hparams.tgt_vocab),
                          batch_size=self.hparams.batch_size, num_workers=0,
                          collate_fn=collate_fn)

    @staticmethod
    def generate_square_subsequent_mask(size):
        return torch.triu(torch.full((size, size), float('-inf')), 1)

def load_data(file_path, max_sequences=1000):
    df = pd.read_csv(file_path, header=None, names=['uniprot_id', 'sequence', 'function', 'pfam_hits'])
    if max_sequences and len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=42)
    sequences = df['sequence'].apply(lambda x: list(x.upper())).tolist()
    descriptions = df['function'].str.split().tolist()
    return sequences, descriptions

if __name__ == "__main__":
    file_path = r"C:\Users\wikik\Documents\mati\Parrot\MODEL\Nightingale\proteins_with_pfam.csv"
    sequences, descriptions = load_data(file_path, max_sequences=2500)
    split = int(0.8 * len(sequences))
    train_seq, val_seq = sequences[:split], sequences[split:]
    train_desc, val_desc = descriptions[:split], descriptions[split:]
    src_vocab = build_vocab([aa for seq in sequences for aa in seq])
    tgt_vocab = build_vocab([tok for desc in descriptions for tok in desc])
    tgt_vocab['<start>'] = len(tgt_vocab)
    tgt_vocab['<end>'] = len(tgt_vocab)

    model = ProteinModel(sequences=train_seq, descriptions=train_desc,
                         sequences_val=val_seq, descriptions_val=val_desc,
                         src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                         src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
                         d_model=256, nhead=8, neuron=1024,
                         num_encoder_layers=2, num_decoder_layers=2,
                         dropout=0.2, lr=1e-4, batch_size=32)

    logger = TensorBoardLogger("tb_logs_proteins", name="protein_model")
    early_stop = EarlyStopping(monitor="val_loss", patience=3)
    trainer = pl.Trainer(logger=logger, callbacks=[early_stop],
                         accelerator="auto", gradient_clip_val=1.0)
    trainer.fit(model)