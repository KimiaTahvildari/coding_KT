import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import importlib.util
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# ======================
# Imports from your repo
# ======================
from preprocessing.segmentation import load_and_preprocess
from preprocessing.windowing import extract_windows
from modeling.model_pytorch import AFCNN
from training.datasets import AFDataset

# ======================
# Paths
# ======================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_ROOT, "ptbxl_database.csv")
SCP_PATH = os.path.join(DATA_ROOT, "scp_statements.csv")

# ======================
# Signal config
# ======================
FS = 500
WINDOW_SEC = 5
WINDOW_SAMPLES = FS * WINDOW_SEC

# ======================
# Training config
# ======================
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Load PTB-XL metadata
# ======================
df = pd.read_csv(CSV_PATH)
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

# ======================
# Load SCP statements
# ======================
scp_df = pd.read_csv(SCP_PATH, index_col=0)

af_codes = scp_df[
    (scp_df["rhythm"] == 1) &
    (scp_df.index.isin(["AFIB", "AFL"]))
].index.tolist()

def has_af(scp_dict):
    return int(any(code in scp_dict for code in af_codes))

df["is_af"] = df["scp_codes"].apply(has_af)

print("AF prevalence:", df["is_af"].mean())

# ======================
# Load signals + windowing
# ======================
X, y = [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading ECGs"):
    record_path = os.path.join(DATA_ROOT, row["filename_hr"])
    ecg = load_and_preprocess(record_path, FS)

    windows, labels = extract_windows(
        ecg,
        label=row["is_af"],
        window_samples=WINDOW_SAMPLES
    )

    X.extend(windows)
    y.extend(labels)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Total windows:", len(y))

# ======================
# Train / validation split
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

train_loader = DataLoader(
    AFDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    AFDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ======================
# Model
# ======================
model = AFCNN().to(DEVICE)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

pos_weight = torch.tensor(weights[1] / weights[0]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Class weights:", weights)

# ======================
# Training loop
# ======================
for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.float().to(DEVICE)

        logits = model(xb).squeeze()
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ---- validation ----
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.extend(probs)
            targets.extend(yb.numpy())

    acc = accuracy_score(targets, np.array(preds) > 0.5)
    auc = roc_auc_score(targets, preds)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss: {np.mean(train_losses):.4f} | "
        f"Acc: {acc:.4f} | "
        f"AUC: {auc:.4f}"
    )
