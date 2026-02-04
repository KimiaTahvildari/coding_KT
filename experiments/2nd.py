# experiments/experiment_2.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------
# Fix project root import
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Imports from your project
# -------------------------------------------------
from preprocessing.segmentation import load_and_preprocess_rr
from preprocessing.windowing import extract_rr_sequences
from training.datasets import RRCNNDataset
from modeling.rr_cnn_lstm import RRCNNLSTM


# -------------------------------------------------
# Configuration
# -------------------------------------------------
DATA_ROOT = r"C:\Users\Kimia\OneDrive - Bahceşehir Üniversitesi\Belgeler\thesis related\coding_KT\data\Ptb-xl"   # <-- CHANGE THIS
METADATA_PATH = r"C:\Users\Kimia\OneDrive - Bahceşehir Üniversitesi\Belgeler\thesis related\coding_KT\data\Ptb-xl\ptbxl_database.csv"
FS = 500                           # PTB-XL sampling rate
SEQ_LEN = 10                       # RR intervals per sample
BATCH_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Load metadata
# -------------------------------------------------
print("Loading PTB-XL metadata...")
df = pd.read_csv(METADATA_PATH)

# -------------------------------------------------
# Binary AF labeling
# -------------------------------------------------
def is_af(label_str):
    return int("AFIB" in label_str)

df["is_af"] = df["scp_codes"].apply(is_af)

print(df["is_af"].value_counts())

# -------------------------------------------------
# RR extraction & windowing
# -------------------------------------------------
X_all, y_all = [], []

print("Extracting RR sequences...")

for idx, row in df.iterrows():
    try:
        record_path = os.path.join(DATA_ROOT, row["filename_hr"])

        rr = load_and_preprocess_rr(record_path, FS)

        if len(rr) < SEQ_LEN:
            continue

        X_seq, y_seq = extract_rr_sequences(
            rr,
            label=row["is_af"],
            seq_len=SEQ_LEN,
            stride=1
        )

        X_all.extend(X_seq)
        y_all.extend(y_seq)

    except Exception as e:
        print(f"Skipping record {idx}: {e}")

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"Total RR samples: {X_all.shape[0]}")
print(f"RR sequence shape: {X_all.shape[1:]}")

# -------------------------------------------------
# Train / Test split (INTER-patient style)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)

# -------------------------------------------------
# Dataset & DataLoader
# -------------------------------------------------
train_dataset = RRCNNDataset(X_train, y_train)
test_dataset = RRCNNDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------------------------------
# Sanity check
# -------------------------------------------------
X_batch, y_batch = next(iter(train_loader))

print("Batch X shape:", X_batch.shape)
print("Batch y shape:", y_batch.shape)

print("Experiment 2 data preparation complete.")

# -------------------------------------------------
# Model, loss, optimizer
# -------------------------------------------------
model = RRCNNLSTM(seq_len=SEQ_LEN).to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "val_auc": []
}

for epoch in range(EPOCHS):
    # ---------- TRAIN ----------
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.float().to(DEVICE)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ---------- VALIDATE ----------
    model.eval()
    val_losses = []
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.float().to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)

            val_losses.append(loss.item())
            preds.extend(probs.cpu().numpy())
            targets.extend(yb.cpu().numpy())

    acc = accuracy_score(targets, np.array(preds) > 0.5)
    auc = roc_auc_score(targets, preds)

    history["train_loss"].append(np.mean(train_losses))
    history["val_loss"].append(np.mean(val_losses))
    history["val_acc"].append(acc)
    history["val_auc"].append(auc)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {history['train_loss'][-1]:.4f} | "
        f"Val Loss: {history['val_loss'][-1]:.4f} | "
        f"Acc: {acc:.4f} | AUC: {auc:.4f}"
    )

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Epoch")
plt.show()

plt.figure()
plt.plot(history["val_acc"], label="Val Accuracy")
plt.plot(history["val_auc"], label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("Validation Performance")
plt.show()
