"""
P3: Norm Classifier — BERT Fine-tuning
=======================================
Loads train/val/test CSVs and fine-tunes bert-base-uncased
for binary classification (norm vs generic).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── 0. CONFIG ────────────────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("\nLoading CSVs...")
df_train = pd.read_csv("train.csv")
df_val   = pd.read_csv("val.csv")
df_test  = pd.read_csv("test.csv")

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# ── 2. DATASET CLASS ─────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class NormDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(NormDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(NormDataset(df_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(NormDataset(df_test),  batch_size=BATCH_SIZE)

# ── 3. MODEL ─────────────────────────────────────────────────────────────────
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# ── 4. TRAIN & EVALUATE FUNCTIONS ────────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return acc, f1, all_preds, all_labels


# ── 5. TRAINING LOOP ─────────────────────────────────────────────────────────
print("\nStarting training...\n")
for epoch in range(EPOCHS):
    train_loss      = train_epoch(model, train_loader)
    val_acc, val_f1, _, _ = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss : {train_loss:.4f}")
    print(f"  Val Acc    : {val_acc:.4f}")
    print(f"  Val F1     : {val_f1:.4f}")
    print()

# ── 6. TEST SET EVALUATION ───────────────────────────────────────────────────
print("Evaluating on test set...")
test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader)

print(f"\n=== FINAL TEST RESULTS ===")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1 Score : {test_f1:.4f}")
print(f"\n=== CLASSIFICATION REPORT ===")
print(classification_report(test_labels, test_preds, target_names=["Generic", "Norm"]))

# ── 7. SAVE MODEL ────────────────────────────────────────────────────────────
model.save_pretrained("norm_classifier_bert")
tokenizer.save_pretrained("norm_classifier_bert")
print("\n✅ Model saved to: norm_classifier_bert/")
