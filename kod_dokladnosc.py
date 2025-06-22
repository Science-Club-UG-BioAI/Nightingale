import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import sacrebleu

# Placeholder for MonMothmaTheGOAT (must be implemented in Venator or replaced with nn.Transformer)
# from Venator import MonMothmaTheGOAT

# Dataset
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
        # Convert sequence to indices
        src = torch.tensor([self.src_vocab.get(aa, self.src_vocab['<pad>']) for aa in self.sequences[idx]], dtype=torch.long)
        # Add <start> and <end> tokens to description
        desc = ['<start>'] + self.descriptions[idx] + ['<end>']
        tgt = torch.tensor([self.tgt_vocab.get(token, self.tgt_vocab['<pad>']) for token in desc], dtype=torch.long)
        return src, tgt, desc

# Collate function
def collate_fn(batch):
    src_batch, tgt_batch, desc_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch, desc_batch

# Lightning Module
class ProteinModel(pl.LightningModule):
    def __init__(self, sequences, descriptions, sequences_val, descriptions_val, src_vocab, tgt_vocab, 
                 src_vocab_size, tgt_vocab_size, d_model, nhead, neuron, num_encoder_layers, num_decoder_layers, 
                 dropout=0.2, lr=0.0001):
        super().__init__()
        self.save_hyperparameters()
        # Placeholder: Replace with MonMothmaTheGOAT or nn.Transformer
        # self.model = MonMothmaTheGOAT(src_vocab_size, d_model, nhead, neuron, num_encoder_layers, num_decoder_layers, tgt_vocab_size, dropout)
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=neuron,
            dropout=dropout,
            batch_first=True
        )
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
        self.validation_outputs = []

    def forward(self, src, tgt, tgt_mask=None):
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)
        output = self.model(src_embed, tgt_embed, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def training_step(self, batch, batch_idx):
        src, tgt, _ = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)
        logits = self(src, tgt_input, tgt_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        non_pad_mask = tgt_output != 0
        correct = (preds == tgt_output) & non_pad_mask
        accuracy = correct.sum().float() / non_pad_mask.sum().float()

        # Log gradient norm
        grad_norm = sum(p.grad.norm() for p in self.parameters() if p.grad is not None)
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train_acc": accuracy}

    def validation_step(self, batch, batch_idx):
        src, tgt, descriptions = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)
        logits = self(src, tgt_input, tgt_mask)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        non_pad_mask = tgt_output != 0
        correct = (preds == tgt_output) & non_pad_mask
        accuracy = correct.sum().float() / non_pad_mask.sum().float()

        # Generate predictions for BLEU
        predicted_ids = logits.argmax(dim=-1)
        predicted_texts = []
        for pred in predicted_ids:
            tokens = [self.inv_tgt_vocab.get(id.item(), "<unk>") for id in pred if id.item() != 0]
            predicted_texts.append(" ".join(tokens))

        self.validation_outputs.append({
            "val_loss": loss,
            "val_acc": accuracy,
            "predicted_texts": predicted_texts,
            "references": [" ".join(desc[1:-1]) for desc in descriptions]  # Exclude <start> and <end>
        })
        return {"val_loss": loss, "val_acc": accuracy}

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in self.validation_outputs]).mean()
        predicted_texts = [text for x in self.validation_outputs for text in x["predicted_texts"]]
        references = [ref for x in self.validation_outputs for ref in x["references"]]

        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(predicted_texts, [references]).score
        self.log("val_bleu", bleu, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True)

        # Print metrics
        train_loss = self.trainer.logged_metrics.get("train_loss_epoch", 0.0)
        train_acc = self.trainer.logged_metrics.get("train_acc_epoch", 0.0)
        print(f"Epoch {self.current_epoch}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, BLEU: {bleu:.4f}")

        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        dataset = ProteinDataset(self.hparams.sequences, self.hparams.descriptions, 
                                self.hparams.src_vocab, self.hparams.tgt_vocab)
        if len(dataset) == 0:
            raise ValueError("Dataset jest pusty. Sprawdź dane wejściowe.")
        return DataLoader(dataset, shuffle=True, num_workers=0, collate_fn=collate_fn)

    def val_dataloader(self):
        dataset = ProteinDataset(self.hparams.sequences_val, self.hparams.descriptions_val, 
                                self.hparams.src_vocab, self.hparams.tgt_vocab)
        if len(dataset) == 0:
            raise ValueError("Dataset walidacyjny jest pusty. Sprawdź dane wejściowe.")
        return DataLoader(dataset, num_workers=0, collate_fn=collate_fn)

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Load data function
def load_data(file_path, max_sequences=1000):
    try:
        df = pd.read_csv(file_path, header=None, names=['uniprot_id', 'sequence', 'function', 'pfam_hits'])
        print(f"Załadowano {len(df)} rekordów.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Plik {file_path} nie istnieje.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Plik {file_path} jest pusty.")

    if max_sequences is not None and len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=42)
    
    sequences = df['sequence'].apply(lambda x: list(x.upper())).tolist()  # Normalize to uppercase
    descriptions = df['function'].str.split().tolist()
    print(f"Przetworzono {len(sequences)} sekwencji.")
    return sequences, descriptions

# Main execution
if __name__ == "__main__":
    # Load data
    file_path = r"C:\Users\wikik\Documents\mati\Parrot\MODEL\Nightingale\proteins_with_pfam.csv"
    max_sequences = 2500
    sequences, descriptions = load_data(file_path, max_sequences)
    
    # Split into training and validation
    if not sequences or not descriptions:
        raise ValueError("Brak danych po załadowaniu pliku. Sprawdź format pliku.")
    split_idx = int(0.8 * len(sequences))
    sequences_train = sequences[:split_idx]
    descriptions_train = descriptions[:split_idx]
    sequences_val = sequences[split_idx:]
    descriptions_val = descriptions[split_idx:]

    # Build vocabularies with special tokens
    all_amino_acids = [aa for seq in sequences for aa in seq]
    all_desc_tokens = [token for desc in descriptions for token in desc]
    src_vocab = {aa: idx + 1 for idx, aa in enumerate(set(all_amino_acids))}
    src_vocab['<pad>'] = 0
    tgt_vocab = {token: idx + 1 for idx, token in enumerate(set(all_desc_tokens))}
    tgt_vocab['<pad>'] = 0
    tgt_vocab['<start>'] = len(tgt_vocab)
    tgt_vocab['<end>'] = len(tgt_vocab)

    # Model parameters
    params = {
        "sequences": sequences_train,
        "descriptions": descriptions_train,
        "sequences_val": sequences_val,
        "descriptions_val": descriptions_val,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "d_model": 256,
        "nhead": 8,
        "neuron": 1024,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dropout": 0.2,
        "lr": 0.0001
    }

    # Initialize model
    model = ProteinModel(**params)

    # Logger and callback
    logger = TensorBoardLogger("tb_logs_proteins", name="protein_model")
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)

    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping],
        accelerator="auto",
        gradient_clip_val=1.0  # Prevent gradient explosion
    )

    # Fit model
    trainer.fit(model)