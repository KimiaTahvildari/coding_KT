# preprocessing/windowing.py

def extract_windows(ecg, label, window_samples):
    windows, labels = [], []
    for i in range(0, ecg.shape[1], window_samples):
        w = ecg[:, i:i + window_samples]
        if w.shape[1] == window_samples:
            windows.append(w)
            labels.append(label)
    return windows, labels
