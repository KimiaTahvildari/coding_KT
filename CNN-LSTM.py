#this is a joint project 
import wfdb
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np
import pywt
import gc
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import psutil
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from src.dataloader import get_dataloaders, compute_class_weights_fast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight



def dwt_denoise(data, wavelet='db4', level=3, threshold_scale=0.5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(data)))
    denoised = [pywt.threshold(c, value=uthresh, mode='soft') if i != 0 else c for i, c in enumerate(coeffs)]
    reconstructed = pywt.waverec(denoised, wavelet)
    print(f"Denoised signal length: {len(reconstructed)}, original: {len(data)}")  
    return reconstructed[:len(data)]  # Trim to original length

def detect_r_peaks_dwt(ecg_signal, fs=500):
    coeffs = pywt.wavedec(ecg_signal, 'db4', level=4)
    d3, d4 = coeffs[3], coeffs[4]
    
    # Truncate to same length
    min_len = min(len(d3), len(d4))
    d3 = d3[:min_len]
    d4 = d4[:min_len]

    qrs_region = d3 + d4

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(qrs_region, distance=int(0.3 * fs))  # 300ms between R-peaks
    return peaks

def butter_highpass_filter(data, cutoff=0.5, fs=500, order=7):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff=40, fs=500, order=6):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)


def load_rhythm_annotations(record_path):
    """
    Loads rhythm annotations from .atr file.
    Returns a list of (sample_index, rhythm_label) tuples.
    """
    ann = wfdb.rdann(record_path, 'atr')
    rhythm_map = {
        '(N': 0,     # Normal
        '(AFIB': 1,  # Atrial Fibrillation
        '(AFL': 2,   # Atrial Flutter
        '(J': 3      # Junctional rhythm
    }
    rhythm_ann = []
    for i, aux in enumerate(ann.aux_note):
        aux = aux.strip()
        if aux in rhythm_map:
            rhythm_ann.append((ann.sample[i], rhythm_map[aux]))
    return rhythm_ann

def label_r_peak(r_idx, rhythm_ann):
    """
    Given an R-peak sample index and sorted rhythm annotations,
    returns the latest rhythm label at or before the R-peak.
    """
    label = 0  # Default: Normal
    for i in range(len(rhythm_ann)):
        if r_idx >= rhythm_ann[i][0]:
            label = rhythm_ann[i][1]
        else:
            break
    return label



def load_ecg(record_path, lead_index=0):
    """Load signal and metadata from a WFDB record"""
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, lead_index]
    return ecg_signal, record.fs  # signal and sampling rate

def summarize_aux_notes(record_path):
    ann = wfdb.rdann(record_path, 'atr')
    aux_notes = [aux.strip() for aux in ann.aux_note]
    return Counter(aux_notes)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data"
RAW_AFDB_DIR = RAW_DATA_DIR / "MIT_AF"
RAW_PTBXL_DIR = RAW_DATA_DIR / "PTB-XL"
PROCESSED_DIR = BASE_DIR / "processed"
NUMPY_DIR = PROCESSED_DIR / "numpy"
PTBXL_NUMPY_DIR = PROCESSED_DIR / "PTB_XL_numpy"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
# NUMPY_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing params
SEGMENT_LENGTH_SEC = 10
LEAD_INDEX = 0
LOWCUT = 0.5
HIGHCUT = 45.0
FILTER_ORDER = 4


RECORDS_FILE = RAW_DATA_DIR / "RECORDS"
NOTES_FILE = RAW_DATA_DIR / "notes.txt"

# Step 1: Load valid record names
with open(RECORDS_FILE, 'r') as f:
    all_records = [line.strip() for line in f.readlines()]

# Step 2: Exclude records listed in notes.txt
with open(NOTES_FILE, 'r') as f:
    notes = [line.strip().split()[0] for line in f if line.strip()]
bad_records = set(notes)
valid_records = [r for r in all_records if r not in bad_records]



summary = Counter()
for record_name in valid_records:
    path = RAW_DATA_DIR / record_name
    summary += summarize_aux_notes(str(path))

print("🔎 Aux note frequency:")
for note, count in summary.items():
    print(f"{note}: {count}")


print(f"✅ {len(valid_records)} valid records will be processed.\n")

# Metadata collection
metadata = []

def process_record(record_name: str, n_augmentations: int = 5):
    record_name = str(record_name).zfill(5)
    record_path = RAW_DATA_DIR / record_name
    try:
        signal, fs = load_ecg(str(record_path), lead_index=LEAD_INDEX)

        # Step 1: Butterworth filtering
        filtered = butter_highpass_filter(signal, cutoff=0.5, fs=fs, order=7)
        filtered = butter_lowpass_filter(filtered, cutoff=40, fs=fs, order=6)

        # Step 2: DWT denoising
        denoised = dwt_denoise(filtered, wavelet='db4', level=3)

        # Step 3: Detect R peaks
        r_peaks = detect_r_peaks_dwt(denoised, fs=fs)

        # Step 4: Annotated AF intervals
        rhythm_ann = load_rhythm_annotations(str(record_path))

        for i in range(len(r_peaks)):
            r1 = r_peaks[i]
            start = max(0, r1 - int(0.25 * fs))
            end = min(len(denoised), r1 + int(0.5 * fs))

            segment = denoised[start:end]
            if len(segment) != int(0.75 * fs):  # Ensure exact 187 samples at 250 Hz
                continue

            label = label_r_peak(r1, rhythm_ann)

            out_name = f"{record_name}_beat{i}"
            out_path = NUMPY_DIR / f"{out_name}.npy"
            os.makedirs(NUMPY_DIR, exist_ok=True)
            np.save(out_path, segment.astype(np.float32))

            metadata.append({
                "record": str(record_name),
                "beat": i,
                "file": out_path.name,
                "af_label": label,
                "fs": fs,
                "r_sample": int(r1)
            })
            

            """# Inside process_record loop
            should_augment = af_label in [1, 2, 3]
            if should_augment:
                for n in range(n_augmentations):
                    aug_segment = rearrange_ecg_segment(segment)
                    out_aug_path = NUMPY_DIR / f"{out_name}_aug{n}.npy"
                    np.save(out_aug_path, aug_segment.astype(np.float32))
                    metadata.append({
                        "record": record_name,
                        "beat": i,
                        "file": out_aug_path.name,
                        "af_label": af_label,
                        "fs": fs,
                        "start_sample": start,
                        "end_sample": end
                    })"""



        print(f"✅ Processed {record_name} → {len(r_peaks) - 1} beats.")
    except Exception as e:
        print(f"❌ Error processing {record_name}: {e}")



for record in valid_records:
    hea_path = RAW_DATA_DIR / f"{record}.hea"
    dat_path = RAW_DATA_DIR / f"{record}.dat"
    qrs_path = RAW_DATA_DIR / f"{record}.qrs"

    if not hea_path.exists() or not dat_path.exists() or not qrs_path.exists():
        print(f"⚠️ Skipping {record} (missing essential files)")
        continue

    process_record(record)

# Save metadata CSV
df = pd.DataFrame(metadata)
df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
print("\nmetadata.csv saved.")
print(f"\nTotal segments processed: {len(metadata)}")



class CNNLSTM_AF(nn.Module):
    def __init__(self, input_length=187, num_classes=4):
        super(CNNLSTM_AF, self).__init__()

        # CNN Block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate resulting length after 3x pooling
        cnn_out_len = input_length // (2 ** 3)  # 187 -> ~23
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * cnn_out_len, 64)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (B, 128, T)
        x = x.permute(0, 2, 1)  # (B, T, Channels) for LSTM
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)



def compute_class_weights_fast(metadata_csv):
    df = pd.read_csv(metadata_csv)
    labels = df['af_label'].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return weights


class ECGDataset(Dataset):
    def __init__(self, metadata_csv, data_dir, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(self.data_dir / row['file']).astype(np.float32).squeeze()
        y = int(row['af_label'])

        # Pad or trim to 187 samples
        desired_length = 187
        if len(x) < desired_length:
            x = np.pad(x, (0, desired_length - len(x)), mode='constant')
        else:
            x = x[:desired_length]

        if self.transform:
            x = self.transform(x)

        # Convert to PyTorch Tensors
        x_tensor = torch.tensor(x).unsqueeze(0)  # (1, length) → channel-first
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor

def get_dataloaders(
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    shuffle=True,
    seed=42,
):
    # Load full dataset
    metadata_csv = PROCESSED_DIR / "metadata.csv"
    dataset = ECGDataset(metadata_csv, PTBXL_NUMPY_DIR)

    # print("Train class distribution:", Counter([dataset[i][1].item() for i in range(len(dataset))]))

    # Split sizes
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    # Fix seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


def track_resources():
    """Returns current CPU, RAM, and GPU usage."""
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().used / (1024 ** 3)
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    return cpu_usage, ram_usage, gpu_memory


def train_model(model, criterion, optimizer, device, epochs, train_loader, val_loader, config_id, patience=5):
    history = defaultdict(list)
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"[{config_id}] Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_accuracy = 100 * train_correct / total
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_accuracy"].append(train_accuracy)

        # Validation
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        true_labels, predictions = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                preds = outputs.argmax(1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(y_batch.cpu().numpy())
                val_correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_accuracy = 100 * val_correct / total
        val_f1 = f1_score(true_labels, predictions, average="macro")
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_accuracy"].append(val_accuracy)

        report = classification_report(true_labels, predictions, output_dict=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(weights_dir, f"{config_id}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"✅ Saved best model: {best_model_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}")

        if patience_counter >= patience:
            print("⏹️ Early stopping.")
            break
        print(f"📊 Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {val_accuracy:.2f}% | F1: {val_f1:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    return history, report, val_f1, best_model_path


def hyperparameter_search(param_grid, train_loader, val_loader, device, model_type, base_config, epochs=20):
    results = []
    best_f1 = 0
    best_params = None

    for config_counter, params in enumerate(itertools.product(*param_grid.values())):
        config_id = f"config_{config_counter}"
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\n🔍 Testing {config_id}: {param_dict}")

        config = base_config.copy()
        config.update(param_dict)
    
        if model_type == "cnnlstm":
            model = CNNLSTM_AF().to(device)
        
            raise ValueError("Unknown model type")

        weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float).to(device)
        criterion = FocalLoss(alpha=weights_tensor[1], gamma=2)        
        optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 0.0005))

        start_time = time.time()
        cpu_start, ram_start, gpu_start = track_resources()

        history, report, val_f1, best_model_path = train_model(
            model, criterion, optimizer, device, epochs, train_loader, val_loader, config_id
        )

        cpu_end, ram_end, gpu_end = track_resources()

        results.append({
            "config_id": config_id,
            "params": config,
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
            "f1_score": val_f1,
            "accuracy": report["accuracy"],
            "saved_model": best_model_path,
            "duration_s": round(time.time() - start_time, 2),
            "cpu_diff": cpu_end - cpu_start,
            "ram_diff": ram_end - ram_start,
            "gpu_diff": gpu_end - gpu_start
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = config

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/hyperparameter_results.csv", index=False)
    print("📝 Saved hyperparameter results to results/hyperparameter_results.csv")
    return best_params, df


# Choose configuration
model_type = "cnnlstm"
epochs = 5000
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")


# Basic Config
base_config = {
    "num_words": 500,
    "embedding_dim": 64,
    "hidden_dim": 64,
    "output_dim": 4,
    "input_dim": 128,
    "dropout": 0.3
}

# Hyperparameter grid
param_grid = {
    "input": [128],
    "dropout": [0.3],
    "lr": [0.001],
    "output_dim": [4]
}


metadata_df = pd.read_csv(PROCESSED_DIR / "metadata.csv")

# DataLoaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
class_weights = compute_class_weights_fast(PROCESSED_DIR / "metadata.csv")

# Start tuning
best_params, results_df = hyperparameter_search(param_grid, train_loader, val_loader, device, model_type, base_config, epochs=epochs)
print(f"🎯 Best config: {best_params}")


""" # Split and balance
af_df = metadata_df[metadata_df["af_label"] == 1]
normal_df = metadata_df[metadata_df["af_label"] == 0].sample(n=len(af_df), random_state=42)

balanced_df = pd.concat([af_df, normal_df])
balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

print(f"Balanced dataset: {balanced_df['af_label'].value_counts().to_dict()}")

balanced_csv_path = PROCESSED_DIR / "balanced_metadata.csv"
balanced_df.to_csv(balanced_csv_path, index=False)

from src.dataloader import ECGDataset
from torch.utils.data import DataLoader, random_split

# Create dataset from new CSV
balanced_dataset = ECGDataset(balanced_csv_path, NUMPY_DIR)

# Split manually
val_size = int(0.2 * len(balanced_dataset))
train_size = len(balanced_dataset) - val_size
train_set, val_set = random_split(balanced_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
"""