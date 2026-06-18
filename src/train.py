import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import math
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer


class Vocabulary:
    def __init__(self, min_freq=2):
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.min_freq = min_freq
        self.freqs = {}

    def build(self, sentences):
        for sentence in sentences:
            for word in sentence.lower().split():
                self.freqs[word] = self.freqs.get(word, 0) + 1
        for word, freq in self.freqs.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence):
        return [self.word2idx.get(w, 1) for w in sentence.lower().split()]

    def __len__(self):
        return len(self.word2idx)



class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_len=50):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]["en"]
        tgt = self.data[idx]["de"]

        src_ids = [2] + self.src_vocab.encode(src)[:self.max_len] + [3]
        tgt_ids = [2] + self.tgt_vocab.encode(tgt)[:self.max_len] + [3]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch



def get_lr(step, d_model, warmup_steps=400):
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)



def train():
   
    print("Loading dataset...")
    dataset = load_dataset("bentrevett/multi30k")
    train_data = dataset["train"]
    valid_data = dataset["validation"]

   
    print("Building vocabularies...")
    src_vocab = Vocabulary(min_freq=2)
    tgt_vocab = Vocabulary(min_freq=2)
    src_vocab.build([x["en"] for x in train_data])
    tgt_vocab.build([x["de"] for x in train_data])
    print(f"src vocab size: {len(src_vocab)}, tgt vocab size: {len(tgt_vocab)}")


    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab)
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


    d_model = 256
    device = torch.device("cpu")
    model = Transformer(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=d_model,
        h=8,
        d_ff=512,
        N=3,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: get_lr(step, d_model, warmup_steps=400)
    )

    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    
    EPOCHS = 10
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

    
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

           
            output = output.reshape(-1, len(tgt_vocab))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in valid_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                output = model(src, tgt_input)
                output = output.reshape(-1, len(tgt_vocab))
                tgt_output = tgt_output.reshape(-1)
                val_loss += criterion(output, tgt_output).item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f"\nEpoch {epoch+1} DONE | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "d_model": d_model
            }, "best_model.pt")
            print(f"  --> saved best model (val loss: {best_val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    train()
