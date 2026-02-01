import os
import ast
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
# Paths
DATA_ROOT = r"C:\Users\Kimia\OneDrive - Bahceşehir Üniversitesi\Belgeler\thesis related\coding_KT\data"

CSV_PATH = r"C:\Users\Kimia\OneDrive - Bahceşehir Üniversitesi\Belgeler\thesis related\coding_KT\data\ptbxl_database.csv"
SCP_PATH = r"C:\Users\Kimia\OneDrive - Bahceşehir Üniversitesi\Belgeler\thesis related\coding_KT\data\scp_statements.csv"

# Signal
FS = 500
RECORD_LEN = 5000        # 10 seconds
WINDOW_SEC = 5
WINDOW_SAMPLES = FS * WINDOW_SEC  # 2500
N_LEADS = 2              # Lead I & II only

# Training
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def low_pass_filter(signal, fs, cutoff=40, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def high_pass_filter(signal, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal)

def band_pass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
def apply_filter_multilead(ecg, fs):
    """
    ecg: (12, 5000)
    returns: filtered ecg (12, 5000)
    """
    filtered = np.zeros_like(ecg)
    for lead in range(ecg.shape[0]):
        filtered[lead] = band_pass_filter(ecg[lead], fs)
    return filtered
def select_lead_I_II(ecg):
    """
    PTB-XL lead order:
    0 = I, 1 = II
    """
    return ecg[[0, 1], :]
def load_and_preprocess(record_path):
    signal, _ = wfdb.rdsamp(record_path)
    ecg = signal.T                        # (12, 5000)

    ecg = apply_filter_multilead(ecg, FS) # noise removal
    ecg = select_lead_I_II(ecg)           # (2, 5000)

    return ecg
def extract_windows(ecg, label):
    """
    ecg: (2, 5000)
    returns: list of (2, 2500) windows
    """
    windows, labels = [], []
    for i in range(0, ecg.shape[1], WINDOW_SAMPLES):
        w = ecg[:, i:i + WINDOW_SAMPLES]
        if w.shape[1] == WINDOW_SAMPLES:
            windows.append(w)
            labels.append(label)
    return windows, labels

# Load main PTB-XL metadata
df = pd.read_csv(CSV_PATH)
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

# Load SCP statements (label definitions)
# Load SCP statements

scp_df = pd.read_csv(SCP_PATH, index_col=0)

# AF-related rhythm codes (from index, not column!)
af_codes = scp_df[
    (scp_df["rhythm"] == 1) &
    (scp_df.index.isin(["AFIB", "AFL"]))
].index.tolist()

# Function to detect AF
def has_af(scp_dict):
    return int(any(code in scp_dict for code in af_codes))

# Apply label
df["is_af"] = df["scp_codes"].apply(has_af)


X, y = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    ecg = load_and_preprocess(os.path.join(DATA_ROOT, row["filename_hr"]))
    windows, labels = extract_windows(ecg, row["is_af"])

    X.extend(windows)
    y.extend(labels)

X = np.array(X, dtype=np.float32)  # (N, 2, 2500)
y = np.array(y, dtype=np.int64)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
class AFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
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
class AFCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)
model = AFCNN().to(DEVICE)
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)
print("Class weights:", weights)
print("AF ratio:", y_train.mean())

pos_weight = torch.tensor(weights[1] / weights[0]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
for epoch in range(EPOCHS):
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
